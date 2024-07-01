import os
import torch
from training.params import parse_args
from open_clip import create_model_and_transforms
from training.utils import ContextualClipV1
import torch.distributed as dist
import re
from training.precision import get_autocast
from tqdm import tqdm
import torch.nn.functional as F
import json
from collections import OrderedDict
from datetime import datetime
import matplotlib.pyplot as plt

import logging


class ZeroShotCOCO:
    def __init__(self, data, args):
        self.nb_concepts = 80
        self.concepts = self.get_concepts()
        self.concepts_embeddings = None
        self.args = args
        self.dataloader = data.dataloader
        self.autocast = get_autocast(args.precision)
        self.device = None

    def get_concepts(self):
        # Get COCO classes
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "training/class_names/coco_things_classes.json",
        )

        with open(path, "r") as f:
            concepts = json.load(f, object_pairs_hook=OrderedDict).values()
        return concepts

    def get_mean_std_entropy(self, model):
        entropy_tensor = torch.zeros(0, device=self.device)
        for i, (images, masks) in enumerate(tqdm(self.dataloader)):
            # if i == 10:
            #     break
            if masks is None:
                continue
            images = images[None, :].to(self.device)
            with self.autocast():
                visual_tokens = model(images, None, None, just_visual_tokens=True)

            similarities = visual_tokens.to(torch.float) @ self.concepts_embeddings.T
            px = similarities[0].softmax(dim=-1)
            entropy = (-px * torch.log(px)).sum(dim=-1)
            entropy_tensor = torch.cat([entropy_tensor, entropy])

        if torch.distributed.is_initialized():
            entropy_tensor = my_all_gather(entropy_tensor)
        # plt.hist(entropy_tensor.cpu().numpy(), bins=100)
        # plt.savefig(os.path.join(self.args.output_dir, 'entropy_distribution.pdf'))
        # plt.close()
        std, mean = torch.std_mean(entropy_tensor)
        return mean, std

    @torch.no_grad()
    def eval(self, model):
        logging.info("starting the evaluation on COCO")
        if torch.distributed.is_initialized():
            self.concepts_embeddings = model.module.embed_words(self.concepts).to(
                torch.float
            )
        else:
            self.concepts_embeddings = model.embed_words(self.concepts).to(torch.float)
        device = next(model.parameters()).device
        self.device = device

        nb_thresholds = 10
        std_multiples = torch.linspace(0.1, 3, steps=nb_thresholds, device=device)
        intersection_cumsum = torch.zeros(
            len(self.concepts) + 1, nb_thresholds + 1, dtype=torch.float, device=device
        )
        union_cumsum = torch.zeros(
            len(self.concepts) + 1, nb_thresholds + 1, dtype=torch.float, device=device
        )

        entropy_with_label_cumsum = torch.zeros(1, dtype=torch.float, device=device)
        with_label_cumsum = torch.zeros(1, dtype=torch.float, device=device)
        entropy_without_label_cumsum = torch.zeros(1, dtype=torch.float, device=device)
        without_label_cumsum = torch.zeros(1, dtype=torch.float, device=device)

        mean_entropy, std_entropy = self.get_mean_std_entropy(model)

        for images, masks in tqdm(self.dataloader):
            # The dataloader gives None when no mask is available
            if masks is None:
                continue
            images = images[None, :].to(device)
            masks = masks[None, :].to(device)

            with self.autocast():
                visual_tokens = model(images, None, None, just_visual_tokens=True)

            masks_full = masks.reshape(*masks.shape[:2], -1)  # [b, 80, nb_token]

            # Compute similarities
            similarities = visual_tokens.to(torch.float) @ self.concepts_embeddings.T

            # Only keep patches which are not background
            patch_has_label_mask = masks_full[0].sum(dim=0) > 0
            px = similarities[0].softmax(dim=-1)
            entropy = (-px * torch.log(px)).sum(dim=-1)

            # Compute avg entropy
            px_with_label = px[patch_has_label_mask]
            px_without_label = px[~patch_has_label_mask]
            entropy_with_label = (
                (-px_with_label * torch.log(px_with_label)).sum(dim=-1).sum()
            )
            entropy_without_label = (
                (-px_without_label * torch.log(px_without_label)).sum(dim=-1).sum()
            )

            # Run different thresholds of entropy / supervised
            intersection = torch.zeros(
                len(self.concepts) + 1,
                nb_thresholds + 1,
                dtype=torch.float,
                device=device,
            )
            union = torch.zeros(
                len(self.concepts) + 1,
                nb_thresholds + 1,
                dtype=torch.float,
                device=device,
            )

            masks_full = torch.cat(
                [masks_full, (~patch_has_label_mask).to(torch.uint8)[None, None]], dim=1
            )  # [b, 81, nb_token]
            for index in range(std_multiples.shape[0] + 1):
                if index == std_multiples.shape[0]:
                    has_low_entropy = patch_has_label_mask
                else:
                    has_low_entropy = (
                        entropy < mean_entropy - std_multiples[index] * std_entropy
                    )

                predictions = similarities.argmax(dim=-1)  # [b, nb_token]
                predictions[0, ~has_low_entropy] = (
                    masks_full.shape[1] - 1
                )  # overwrite predictions with label 80 (background) if the entropy is high

                predictions_one_hot = torch.zeros_like(masks_full)
                predictions_one_hot.scatter_(
                    dim=1,
                    index=predictions.unsqueeze(1),
                    src=torch.ones_like(
                        predictions.unsqueeze(1), dtype=predictions_one_hot.dtype
                    ),
                )

                # Compute intersection and union (sum over patch dimension and batch dimension)
                intersection[:, index] = (
                    torch.logical_and(predictions_one_hot, masks_full)
                    .sum(dim=-1)
                    .sum(dim=0)
                    .float()
                )
                union[:, index] = (
                    torch.logical_or(predictions_one_hot, masks_full)
                    .sum(dim=-1)
                    .sum(dim=0)
                    .float()
                )

            # Update cumsums
            intersection_cumsum += intersection
            union_cumsum += union

            entropy_with_label_cumsum += entropy_with_label
            entropy_without_label_cumsum += entropy_without_label
            with_label_cumsum += px_with_label.shape[0]
            without_label_cumsum += px_without_label.shape[0]

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            dist.all_reduce(intersection_cumsum)
            dist.all_reduce(union_cumsum)

            dist.all_reduce(entropy_with_label_cumsum)
            dist.all_reduce(entropy_without_label_cumsum)
            dist.all_reduce(with_label_cumsum)
            dist.all_reduce(without_label_cumsum)

        logging.info("done with the evaluation on COCO")

        miou_score = intersection_cumsum / union_cumsum
        avg_entropy_with_label = entropy_with_label_cumsum / with_label_cumsum
        avg_entropy_without_label = entropy_without_label_cumsum / without_label_cumsum
        return (
            100 * miou_score.mean(dim=0),
            avg_entropy_with_label.item(),
            avg_entropy_without_label.item(),
            mean_entropy.item(),
            std_entropy.item(),
        )


def gather_all_gpus_scalar(local_tensor):
    feats_all = torch.empty(
        dist.get_world_size(),
        *local_tensor.shape,
        dtype=local_tensor.dtype,
        device=local_tensor.device
    )
    output_l = list(feats_all.unbind(0))
    output_all_reduce = torch.distributed.all_gather(
        output_l, local_tensor, async_op=True
    )
    output_all_reduce.wait()
    return torch.stack(output_l)


def my_all_gather(q):
    ws = int(os.environ["WORLD_SIZE"])
    local_size = torch.tensor(q.shape[0], device=q.device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(ws)]
    dist.all_gather(all_sizes, local_size)
    max_size = max(all_sizes)

    size_diff = max_size - q.shape[0]
    if size_diff > 0:
        padding = torch.zeros(size_diff, device=q.device, dtype=q.dtype)
        q = torch.cat((q, padding))

    all_qs_padded = [torch.zeros_like(q) for _ in range(ws)]
    dist.all_gather(all_qs_padded, q)
    output_l = []
    for q, size in zip(all_qs_padded, all_sizes):
        output_l.append(q[:size])
    return torch.cat(output_l, dim=0)


if __name__ == "__main__":
    import sys
    import numpy as np

    args = parse_args(sys.argv[1:])
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device="cuda",
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        pretrained_image=args.pretrained_image,
        image_mean=args.image_mean,
        image_std=args.image_std,
        aug_cfg=args.aug_cfg,
        output_dict=True,
    )
    model_kwargs = {}
    model_kwargs["init_logit_scale"] = np.log(10)  # different from CLIP
    model_kwargs["init_logit_bias"] = -10
    model = ContextualClipV1(model, args, "cuda", **model_kwargs)

    torch.distributed.init_process_group(
        backend="nccl",
        init_method="file:///home/tim/Documents/KU_LEUVEN/repos/Contextual-CLIP/init",
        world_size=1,
        rank=0,
    )
    root_dir = "/mnt/mp600_2Tb/KU_LEUVEN/coco"
    patch_size = 32

    zeroshot_coco = ZeroShotCOCO(root_dir, patch_size, args)

    zeroshot_coco.eval(model)  # TODO: which model is the right one?
