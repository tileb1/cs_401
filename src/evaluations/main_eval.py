# ---------------------------------------------------------------------------------------------------
# CLIP-DINOiser
# authors: Monika Wysoczanska, Warsaw University of Technology
# ---------------------------------------------------------------------------------------------------
# modified from TCL (https://github.com/kakaobrain/tcl/) Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ---------------------------------------------------------------------------------------------------

import os
import argparse
import stat
import datasets.transforms
import sys
import json

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from helpers.dist_utils import init_distributed_mode
from hydra import compose, initialize
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, set_random_seed

# from mmseg.apis import multi_gpu_test

from helpers.logger import get_logger
from models import build_model
from segmentation.evaluation import (
    build_seg_dataloader,
    build_seg_dataset,
    build_seg_inference,
)
from open_clip.factory import get_model_config
from omegaconf import OmegaConf, open_dict

# from training.my_utils.run_manager import bool_flag
from training.distributed import init_distributed_device

# from training.wandb_log import update_wandb_run_with_raw_metadata
from evaluations.my_multi_gpu_test import multi_gpu_test


@torch.no_grad()
def evaluate(cfg, val_loaders, args, device, checkpoint_path):
    logger = get_logger()
    ret = {}

    for key, loader in val_loaders.items():

        logger.info(f"### Validation dataset: {key}")
        CLASSES = loader.dataset.CLASSES
        logger.info(f"Creating model:{cfg.model.type}")

        # Load the config
        model_cfg = get_model_config(args.model)
        model_cfg["type"] = cfg.model["type"]
        model_cfg["model_name"] = args.model
        model_cfg["visual_norm"] = False
        model_cfg["use_templates"] = cfg.model["use_templates"]
        del model_cfg["custom_text"]
        model_cfg = OmegaConf.create(model_cfg)
        model = build_model(model_cfg, class_names=CLASSES)
        model.handle_bkg = False
        if key in ["voc", "coco_object", "context"]:
            model.handle_bkg = True

        state_dict = torch.load(checkpoint_path)["state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.device = device
        model.eval()

        # Embed the classes
        model.register_class_embeddings()

        metrics = validate_seg(cfg, cfg.evaluate.get(key), loader, model, args)
        if isinstance(metrics, dict):
            threshold_dict = {}
            for threshold, metric in metrics.items():
                miou = metric["mIoU"] * 100
                logger.info(
                    f"[{key}] mIoU of {len(loader.dataset)} test images: {miou:.2f}%"
                )
                threshold_dict[threshold] = miou
                ret[f"{key}_miou"] = miou
            ret[f"{key}_miou"] = threshold_dict
        else:
            miou = metrics[0]["mIoU"] * 100
            ret[f"{key}_miou"] = miou

    ret["avg_miou_foreground"] = np.mean(
        [v for k, v in ret.items() if "miou" in k and not isinstance(v, dict)]
    )
    ret["avg_miou_background"] = np.mean(
        [max(v.values()) for k, v in ret.items() if "miou" in k and isinstance(v, dict)]
    )
    return ret


@torch.no_grad()
def validate_seg(config, seg_config, data_loader, model, args):
    logger = get_logger()
    dist.barrier()
    model.eval()
    seg_model = build_seg_inference(
        model,
        data_loader.dataset,
        config,
        seg_config,
    )

    # Disable gradients
    for p in seg_model.parameters():
        p.requires_grad = False

    mmddp_model = seg_model
    mmddp_model.eval()

    results = multi_gpu_test(
        model=mmddp_model,
        data_loader=data_loader,
        tmpdir=None,
        gpu_collect=True,
        efficient_test=False,
        pre_eval=True,
        format_only=False,
    )

    if isinstance(results, dict):
        metric = [
            data_loader.dataset.evaluate(
                threshold_results, metric="mIoU", logger=logger
            )
            for threshold_results in results.values()
        ]
    else:
        metric = [data_loader.dataset.evaluate(results, metric="mIoU", logger=logger)]

    if isinstance(results, dict):
        metric = {k: v for k, v in zip(results.keys(), metric)}
    return metric


def main(args):
    # Load the config
    initialize(config_path="configs", version_base=None)
    cfg = compose(config_name=args.config)

    # fully initialize distributed device environment
    device = init_distributed_device(args)
    # mp.set_start_method("fork", force=True)
    # init_dist("pytorch")
    rank, world_size = get_dist_info()
    print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")

    dist.barrier()
    set_random_seed(cfg.seed, use_rank_shift=True)
    cudnn.benchmark = True

    logger = get_logger(cfg)

    val_loaders = {}
    for key in cfg.evaluate.task:
        loader = build_seg_dataloader(
            build_seg_dataset(cfg.evaluate.get(key), args.untar_path, args.local_rank)
        )
        val_loaders[key] = loader

    if args.checkpoint_path.endswith(".pt"):
        checkpoint_paths = [args.checkpoint_path]
        ROOT = os.path.dirname(args.checkpoint_path)
    else:
        checkpoint_paths = [
            os.path.join(args.checkpoint_path, i)
            for i in os.listdir(args.checkpoint_path)
            if i.endswith(".pt")
        ]
        ROOT = args.checkpoint_path

    EVAL_STRUCT = {}
    for checkpoint_path in checkpoint_paths:
        res = evaluate(cfg, val_loaders, args, device, checkpoint_path)
        logger.info(res)
        print(type(res))
        print(res)
        EVAL_STRUCT[checkpoint_path.split("/")[-1]] = res

    with open(os.path.join(ROOT, "results_custom_eval.json"), "w") as f:
        json.dump(EVAL_STRUCT, f)

    with open(os.path.join(ROOT, "results_custom_eval.json"), "r") as f:
        EVAL_STRUCT = json.load(f)

    # if args.rank == 0:
    #     update_wandb_run_with_raw_metadata(str(os.path.dirname(ROOT)).split('/')[-1], EVAL_STRUCT, args)

    dist.barrier()
    dist.destroy_process_group()
    sys.exit(0)


def parse_args():
    parser = argparse.ArgumentParser(description="mmseg test (and eval) a model")
    parser.add_argument("--config", default="ours.yaml", help="config file path")

    # Added by us
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="""url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html""",
    )
    parser.add_argument(
        "--local_rank",
        default=0,
        type=int,
        help="Please ignore and do not set this argument.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ours_dinov2_ViT-B-14_reg",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/scratch/project_465000727/repos/Contextual-CLIP/log_cclip/240424_test_denoiser_transformer_dtemp05_pos1/checkpoints/epoch_8.pt",
        help="Path to checkpoint.",
    )
    parser.add_argument(
        "--untar_path",
        type=str,
        default="/dev/shm",
        help="Path where to untar.",
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
