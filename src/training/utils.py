import os
import math
from einops import rearrange, repeat, reduce
import torch
from torch import nn
import open_clip
import torch.nn.functional as F
import warnings
import numpy as np
from training.precision import get_autocast
import torch.distributed as dist
from datetime import datetime
import torchvision
import matplotlib.pyplot as plt
from datetime import datetime


class MMAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.weights_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.weights_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.weights_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        # Get head-wise representations
        q = rearrange(self.weights_q(q), "b l (h d) -> b l h d", h=self.num_heads)
        k = rearrange(self.weights_k(k), "n (h d) -> n h d", h=self.num_heads)
        v = rearrange(self.weights_v(v), "n (h d) -> n h d", h=self.num_heads)

        # Compute the self-attention
        attn = torch.einsum("b l h d, n h d -> b l h n", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Update the values
        v = torch.einsum("b l h n, n h d -> b l h d", attn, v)
        v = rearrange(v, "b l h d -> b l (h d)")
        v = self.proj(v)
        v = self.proj_drop(v)
        return v, attn


class Clustering:
    def __init__(self, args):
        # self.patch_size = args.patch_size
        # self.assignment_type = args.assignment_type
        # self.n_tokens = args.n_tokens
        # self.pos_alpha = args.pos_alpha
        # self.sinkhorn_lambda = args.sinkhorn_lambda
        # self.sinkhorn_iterations = args.sinkhorn_iterations
        self.args = args

    @torch.no_grad()
    def sinkhorn(self, M, r, c, lambda_, iterations):
        P = torch.exp(-lambda_ * M).float()
        P /= reduce(P, "b n k -> b 1 1", reduction="sum")

        # Iterate over the sinkhorn algorithm
        for _ in range(iterations):
            u = reduce(P, "b n k -> b n 1", reduction="sum")
            P *= r / u
            u = reduce(P, "b n k -> b 1 k", reduction="sum")
            P *= c / u
        P = torch.nan_to_num(P, nan=1e-8)
        return P, torch.sum(P * M, dim=[1, 2])

    def compute_assignments(
        self, tokens, positions, k, pos_alpha, sinkhorn_lambda, sinkhorn_iterations
    ):
        # Normalize the tokens
        tokens = F.normalize(tokens, dim=-1)  # TODO: check if in-place.

        # Get the dimensions
        b, n, d = tokens.shape

        # Compute the random distribution
        r_uni = torch.ones([b, n, 1], device=self.args.device) / n
        r = r_uni

        c = torch.ones([b, 1, k], device=self.args.device) / k

        # Get the indices as one-hot
        p = r_uni.squeeze()
        index = p.multinomial(num_samples=k, replacement=False)
        index = rearrange(index, "b k -> (b k)")
        index = torch.eye(n, device=index.device)[index].to(tokens.device)
        index = rearrange(index, "(b k) n -> b k n", b=b)

        # Set the initial centroids
        centroids = torch.einsum("b n d, b k n -> b k d", tokens, index)

        assignment = index.permute(0, 2, 1)

        for _ in range(self.args.clustering_n_iter):
            # Compute the semantic similarity
            sem_similarity = torch.einsum("b n d, b k d -> b n k", tokens, centroids)

            # Compute the distance matrix
            pos_similarity = torch.sqrt(
                torch.sum(
                    (positions[:, None, :, :] - positions[:, :, None, :]) ** 2, dim=-1
                )
            )
            pos_similarity = torch.einsum(
                "B N n, B n k -> B N n k", pos_similarity, assignment
            )

            tmp = torch.ones_like(pos_similarity)
            tmp[pos_similarity == 0.0] = 0.0
            tmp = tmp.sum(dim=2, keepdim=True)
            pos_similarity[torch.logical_and(pos_similarity == 0.0, tmp != 0.0)] = (
                1e4  # If column is not zero, replace all 0 values with high value
            )
            pos_similarity = reduce(pos_similarity, "B N n k -> B N k", reduction="min")

            # If cost is 0, replace with average cost
            avg_cost = pos_similarity.mean(dim=[1, 2], keepdim=True)
            avg_cost = repeat(avg_cost, "b 1 1 -> b n k", k=k, n=n)
            pos_similarity[pos_similarity == 0.0] = avg_cost[pos_similarity == 0.0]

            pos_similarity /= pos_similarity.amax(dim=(-1, -2))[:, None, None]

            # Get the cost
            M = -sem_similarity + pos_alpha * pos_similarity
            M = (M - M.min()) / (M.max() - M.min())

            # Compute the transportation plan and the distance
            assignment, cost = self.sinkhorn(
                M=M, r=r, c=c, lambda_=sinkhorn_lambda, iterations=sinkhorn_iterations
            )

            # Compute the hard assignments
            hard_assignment = torch.max(assignment, dim=-1, keepdim=True).values
            hard_assignment = repeat(hard_assignment, "b n 1 -> b n k", k=k)
            hard_assignment = (assignment == hard_assignment).float()
            assignment = hard_assignment

            # Update c
            c = hard_assignment.sum(dim=1, keepdim=True) + 1e-2
            c /= c.sum(dim=-1, keepdim=True)

            # Update the centroids
            centroids = torch.einsum("b n d, b n k -> b k d", tokens, assignment)
            centroids = F.normalize(centroids, dim=-1)

        # Normalize column-wise and view-wise
        assignment = rearrange(assignment, "b (m n) k -> m b n k", m=2)
        assignment_v1, assignment_v2 = assignment.unbind()

        # Normalize hard assignment
        # If a cluster is not present in two views, the normalization will divide by 0
        # If that happens, we just replace the 0 by 1
        # Later on, the centroids originating from that cluster will be discarded anyways
        tmpv1 = assignment_v1.sum(dim=-2, keepdim=True)
        tmpv2 = assignment_v2.sum(dim=-2, keepdim=True)
        tmpv1[tmpv1 == 0.0] = 1.0
        tmpv2[tmpv2 == 0.0] = 1.0

        assignment_v1 = assignment_v1 / tmpv1
        assignment_v2 = assignment_v2 / tmpv2
        assignment = torch.cat([assignment_v1, assignment_v2], dim=1)
        return assignment, cost, index

    def compute_centroids(self, tokens, assignments):
        # Compute the centroids of each view and normalize the assignments
        tokens = rearrange(tokens, "(v b) n d -> v b n d", v=2)
        assignments = rearrange(assignments, "b (v n) k -> b v n k", v=2)
        centroids = torch.einsum("v b n d, b v n k -> v b k d", tokens, assignments)
        centroids = rearrange(centroids, "v b k d -> v (b k) d")
        centroids_v1, centroids_v2 = centroids.unbind()

        # Discard a cluster if it's empty in either view
        assignments_v1, assignments_v2 = rearrange(
            assignments, "b v n k -> v (b k) n"
        ).unbind()
        valid_centroids = torch.logical_and(
            (assignments_v1.sum(dim=-1) > 0), (assignments_v2.sum(dim=-1) > 0)
        )
        centroids_v1, centroids_v2 = (
            centroids_v1[valid_centroids],
            centroids_v2[valid_centroids],
        )

        # Correct the number of centroids per view
        centroids_per_view = torch.tensor_split(valid_centroids, tokens.shape[0])
        centroids_per_view = torch.stack([cpv.sum() for cpv in centroids_per_view])

        # Count the average number of regions
        region_count = centroids_per_view.float().mean().item()
        return (
            (centroids_v1, centroids_v2),
            valid_centroids,
            assignments,
            region_count,
            centroids_per_view,
        )

    @torch.no_grad()
    def get_assignments(self, tokens, positions):
        # Discard the [CLS] token
        tokens = rearrange(tokens, "(v b) n d -> b (v n) d", v=2)

        # Patchify positional encodings
        positions = rearrange(positions, "v b d n -> b (v n) d")

        # Compute the assignments
        assignments, _, _ = self.compute_assignments(
            tokens,
            positions,
            self.args.n_tokens,
            self.args.pos_alpha,
            self.args.sinkhorn_lambda,
            self.args.sinkhorn_iterations,
        )
        return assignments


class ContextualClipV1(nn.Module):
    def __init__(
        self,
        model,
        args,
        device,
        init_logit_scale=np.log(1 / 0.07),
        init_logit_bias=None,
    ):
        super().__init__()
        self.model = model
        self.model.visual.output_tokens = True
        self.args = args
        self.device = device
        self.autocast = get_autocast(args.precision)

        # Attributes will be set after the model is on the device
        self.concepts = None
        self.concepts_embeddings = None
        self.concepts_embeddings_normed = None
        self.ln_post = None
        self.vision_to_text_proj = None
        self.attention_layer = None
        self.visual_concepts = None
        self.tokenizer = open_clip.get_tokenizer(self.args.model)

        # self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        # if init_logit_bias is not None:
        #     self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        # else:
        #     self.logit_bias = None
        # self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        # self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)

        if isinstance(device, str):
            device = torch.device(device)

        self.to(device=device)

        # Initialize the visual concepts
        self.visual_concepts = torch.load(self.args.visual_concepts_init_path).to(
            self.device
        )
        self.visual_concepts = F.normalize(self.visual_concepts, p=2, dim=-1)
        print(
            "Using visual concepts at the following path: {}".format(
                self.args.visual_concepts_init_path
            )
        )

        self.get_concepts()
        self.embed_concepts()
        self.concepts = np.array(self.concepts)

        self.plot_counter = 0
        self.plot_counter_max = 10

    def get_concepts(self):
        # Get all the class files
        # concepts = []
        with open(self.args.concepts_path, "r") as f:
            self.concepts = [c.strip() for c in f.readlines()]
            self.concepts_tokens = self.tokenizer(self.concepts).to(self.device)

    @torch.no_grad()
    def embed_words(self, words):
        tokens = self.tokenizer(words).to(self.device)

        # Feed to the text encoder
        chunk_size = 128
        n_chunks = tokens.shape[0] // chunk_size + 1
        words_embeddings = []
        for chunk in torch.chunk(tokens, chunks=n_chunks, dim=0):
            with self.autocast():
                with torch.no_grad():
                    words_embeddings.append(self.model.encode_text(chunk))
        words_embeddings = torch.cat(words_embeddings, dim=0)
        print("Successfully computed the class embeddings.")
        return words_embeddings

    @torch.no_grad()
    def embed_concepts(self):
        with torch.no_grad():
            with self.autocast():
                concepts_embeddings = self.model.encode_text(self.concepts_tokens)

        self.concepts_embeddings = concepts_embeddings
        self.concepts_embeddings_normed = F.normalize(concepts_embeddings, p=2, dim=-1)

    def sync_concepts(
        self,
    ):
        if self.args.distributed:
            # All-reduce the tensor across all GPUs to compute the average
            dist.all_reduce(self.visual_concepts)
            self.visual_concepts = F.normalize(self.visual_concepts, dim=-1, p=2)

    @torch.no_grad()
    def compute_assignments(self, tokens, topk_indices):
        # Retrieve the corresponding concepts
        # topk_indices is b,k
        visual_concepts = self.visual_concepts[topk_indices]

        # TODO: below is a complicated way to initialize non-initialized centroids
        """
        uninitialized = torch.count_nonzero(visual_concepts, dim=-1) == 0

        # Compute indices
        unique_indices, inverse_indices = topk_indices.unique(return_inverse=True) # works on flattened topk_indices

        # Average the [CLS] of the images that triggered the same concept
        b, k = topk_indices.shape
        d = tokens.shape[-1]
        local_initialization = repeat(tokens[:, 0][:, None], 'b 1 d -> (b k) d', k=k)
        pooled_initialization = torch.zeros([unique_indices.shape[0], d], device=tokens.device, dtype=tokens.dtype)
        pooled_initialization.scatter_reduce_(
            dim=0,
            index=repeat(inverse_indices[..., None], 'b k 1 -> (b k) d', d=d),
            src=local_initialization,
            reduce='mean'
        )

        # Expand back the averaged [CLS]
        local_initialization = pooled_initialization.gather(
            dim=0,
            index=repeat(inverse_indices[..., None], 'b k 1 -> (b k) d', d=d),
        )
        local_initialization = rearrange(local_initialization, '(b k) d -> b k d', b=b)

        # Initialize the visual concepts
        visual_concepts[uninitialized] = local_initialization[uninitialized]
        """

        # Normalize the visual concepts
        visual_concepts = F.normalize(visual_concepts, dim=-1, p=2)  # b,k,2,d

        # Get the generic background
        # bg_concept = self.visual_concepts[-1]
        # if torch.count_nonzero(bg_concept) == 0:
        #     bg_concept = F.normalize(tokens[:, 1:].mean(dim=(0, 1)), dim=-1, p=2)

        # Stack the background
        # b, k, d = visual_concepts.shape
        # centroids = torch.stack([visual_concepts, bg_concept[None, None].expand(b, k, -1)], dim=2)
        centroids = visual_concepts

        # Discard the [CLS]
        tokens = tokens[:, 1:]

        # K-means TODO: check norm
        for _ in range(self.args.kmeans_iters):
            centroids_similarity = torch.einsum(
                "b n d, b k c d -> b n k c", tokens, centroids
            )
            assignments = centroids_similarity.argmax(dim=-1).float()

            # Background centroids
            centroids[:, :, 1, :] = torch.einsum(
                "b n d, b n k -> b k d", tokens, assignments
            )

            # Foreground centroids
            centroids[:, :, 0, :] = torch.einsum(
                "b n d, b n k -> b k d", tokens, 1.0 - assignments
            )

            # Normalize
            centroids = F.normalize(centroids, dim=-1, p=2)

        # Update concepts
        concepts_ema = (
            self.args.beta_ema * visual_concepts
            + (1.0 - self.args.beta_ema) * centroids
        )
        self.visual_concepts.scatter_reduce_(
            dim=0,
            index=rearrange(topk_indices, "b k -> (b k) 1 1"),
            src=rearrange(concepts_ema, "b k c d -> (b k) c d"),
            reduce="sum",
        )

        # Normalize
        self.visual_concepts = F.normalize(self.visual_concepts, dim=-1, p=2)
        return assignments

    def forward_eval(self, images):
        _, visual_tokens = self.model.visual(images)

        # Project to the text space
        visual_tokens = self.ln_post(visual_tokens)
        visual_tokens = (
            visual_tokens @ self.vision_to_text_proj
        )  # TODO check the norm issue
        visual_tokens = F.normalize(visual_tokens, dim=-1, p=2)
        return visual_tokens

    def compute_weighted_tokens(self, tokens, topk_indices):
        visual_concepts = self.visual_concepts[topk_indices]  # b,k,d
        similarities = torch.einsum(
            "b n d, b k d -> b k n", tokens[:, 1:], visual_concepts
        )  # ---------------------------- check
        similarities = torch.stack([similarities, -similarities], dim=0)
        similarities = F.softmax(similarities, dim=-1)
        weighted_tokens = torch.einsum(
            "b n d, s b k n -> s b k d", tokens[:, 1:], similarities
        )  # ---------------------------- check
        weighted_tokens = F.normalize(weighted_tokens, dim=-1, p=2)

        # Update concepts
        # with torch.no_grad():
        #     concepts_ema = self.args.beta_ema * visual_concepts + (1. - self.args.beta_ema) * weighted_tokens.detach()
        #     self.visual_concepts[:, 0].scatter_reduce_(
        #         dim=0,
        #         index=rearrange(topk_indices, 'b k -> (b k) 1'),
        #         src=rearrange(concepts_ema, 'b k d -> (b k) d'),
        #         reduce='sum'
        #     )
        # self.visual_concepts = F.normalize(self.visual_concepts, dim=-1, p=2)
        return weighted_tokens

    def plot_tmp(self, image, topk_indices, visual_tokens, text_features, raw_text):
        if self.plot_counter == 0:
            self.plot_counter = self.plot_counter_max
            if self.args.rank in [0]:
                mean = torch.tensor((0.48145466, 0.4578275, 0.40821073)).cuda()[
                    :, None, None
                ]
                std = torch.tensor((0.26862954, 0.26130258, 0.27577711)).cuda()[
                    :, None, None
                ]
                image = ((image * std) + mean) * 255
                image = image.to(torch.uint8)

                self.embed_concepts()
                selected_concept_indices = topk_indices[0]
                selected_vision_concepts = self.visual_concepts[
                    selected_concept_indices
                ]
                OUTPUT_RES = 16
                PATCH_SIZE = 14
                spatial_predictions = torch.einsum(
                    "n d, k d -> n k", visual_tokens[0, 1:], selected_vision_concepts
                )
                spatial_predictions = rearrange(
                    spatial_predictions,
                    "(h w) k -> k 1 h w",
                    h=OUTPUT_RES,
                    w=OUTPUT_RES,
                )
                spatial_predictions = (
                    F.interpolate(
                        spatial_predictions.float(),
                        scale_factor=PATCH_SIZE,
                        mode="bicubic",
                    )
                    .squeeze(1)
                    .cpu()
                )

                selected_textual_concepts = self.concepts_embeddings_normed[
                    selected_concept_indices
                ]
                spatial_predictions2 = torch.einsum(
                    "n d, k d -> n k", visual_tokens[0, 1:], selected_textual_concepts
                )
                spatial_predictions2 = rearrange(
                    spatial_predictions2,
                    "(h w) k -> k 1 h w",
                    h=OUTPUT_RES,
                    w=OUTPUT_RES,
                )
                spatial_predictions2 = (
                    F.interpolate(
                        spatial_predictions2.float(),
                        scale_factor=PATCH_SIZE,
                        mode="bicubic",
                    )
                    .squeeze(1)
                    .cpu()
                )

                spatial_predictions3 = torch.einsum(
                    "n d, d -> n", visual_tokens[0, 1:], text_features[0]
                )
                spatial_predictions3 = rearrange(
                    spatial_predictions3, "(h w) -> 1 1 h w", h=OUTPUT_RES, w=OUTPUT_RES
                )
                spatial_predictions3 = (
                    F.interpolate(
                        spatial_predictions3.float(),
                        scale_factor=PATCH_SIZE,
                        mode="bicubic",
                    )
                    .squeeze()
                    .cpu()
                )

                spatial_predictions4 = torch.einsum(
                    "n d, d -> n", visual_tokens[0, 1:], visual_tokens[0, 0]
                )
                spatial_predictions4 = rearrange(
                    spatial_predictions4, "(h w) -> 1 1 h w", h=OUTPUT_RES, w=OUTPUT_RES
                )
                spatial_predictions4 = (
                    F.interpolate(
                        spatial_predictions4.float(),
                        scale_factor=PATCH_SIZE,
                        mode="bicubic",
                    )
                    .squeeze()
                    .cpu()
                )

                fig, axs = plt.subplots(
                    nrows=4,
                    ncols=self.args.n_tokens + 1,
                    figsize=(self.args.n_tokens * 5 + 5, 10),
                )
                for i in range(self.args.n_tokens):
                    axs[0][i].imshow(image.cpu().permute(1, 2, 0), alpha=0.5)
                    preds_show = axs[0][i].imshow(
                        spatial_predictions[i].detach(), cmap="magma", alpha=0.6
                    )
                    axs[0][i].set_title(self.concepts[topk_indices[0][i].item()])
                    plt.colorbar(preds_show, ax=axs[0][i])

                    axs[1][i].imshow(image.cpu().permute(1, 2, 0), alpha=0.5)
                    preds_show2 = axs[1][i].imshow(
                        spatial_predictions2[i].detach(), cmap="magma", alpha=0.6
                    )
                    axs[1][i].set_title(self.concepts[topk_indices[0][i].item()])
                    plt.colorbar(preds_show2, ax=axs[1][i])

                    axs[2][i].imshow(image.cpu().permute(1, 2, 0), alpha=0.5)
                    axs[3][i].imshow(image.cpu().permute(1, 2, 0), alpha=0.5)

                axs[0][self.args.n_tokens].imshow(image.cpu().permute(1, 2, 0))
                axs[0][self.args.n_tokens].set_title("Visual")
                axs[1][self.args.n_tokens].imshow(image.cpu().permute(1, 2, 0))
                axs[1][self.args.n_tokens].set_title("Textual")

                axs[2][self.args.n_tokens].imshow(image.cpu().permute(1, 2, 0))
                axs[2][self.args.n_tokens].set_title(raw_text[0])
                preds_show3 = axs[2][0].imshow(
                    spatial_predictions3.detach(), cmap="magma", alpha=0.6
                )
                plt.colorbar(preds_show3, ax=axs[2][0])

                axs[3][self.args.n_tokens].imshow(image.cpu().permute(1, 2, 0))
                axs[3][self.args.n_tokens].set_title("Visual CLS")
                preds_show4 = axs[3][0].imshow(
                    spatial_predictions4.detach(), cmap="magma", alpha=0.6
                )
                plt.colorbar(preds_show4, ax=axs[3][0])
                output_file = os.path.join(
                    self.args.logs,
                    self.args.name,
                    "output_images",
                    str(self.args.rank),
                    str(datetime.now()).replace(".", "_"),
                )
                plt.savefig(output_file)
        else:
            self.plot_counter -= 1

    def forward(self, images, text, raw_text, just_visual_tokens=False, eval=False):
        if eval:
            return self.forward_eval(images)

        # Recompute textual concepts
        # self.embed_concepts()

        # Get the visual representations
        visual_tokens, cls_tokens = self.model.visual.trunk.get_intermediate_layers(
            images, return_prefix_tokens=True, norm=False
        )[
            0
        ]  # Keep in mind: last layer is the layer norm (self.model.visual.norm)
        cls_tokens = cls_tokens[:, 0].unsqueeze(1)
        visual_tokens = torch.cat([cls_tokens, visual_tokens], dim=1)
        visual_tokens = F.normalize(visual_tokens, dim=-1, p=2)

        if just_visual_tokens:
            return visual_tokens

        # Get text representations
        text_features = self.model.encode_text(text, normalize=True)

        # Get image features
        if self.args.image_features_type == "cls":
            image_features = visual_tokens[:, 0]
        elif self.args.image_features_type == "pacl":
            texttopatch = torch.einsum(
                "b d, b n d -> b n", text_features, visual_tokens[:, 1:]
            )
            texttopatch = F.softmax(texttopatch, dim=-1)
            image_features = torch.einsum(
                "b n d, b n -> b d", visual_tokens[:, 1:], texttopatch
            )
            image_features = F.normalize(image_features, dim=-1, p=2)
        else:
            raise NotImplementedError

        # Find the concepts represented in the image with the [CLS] tokens
        if self.args.match_type == "textual":
            similarities = image_features @ self.concepts_embeddings_normed.T
        elif self.args.match_type == "visual":
            similarities = image_features @ self.visual_concepts.T
        else:
            raise NotImplementedError
        topk_indices = torch.topk(similarities, k=self.args.n_tokens, dim=-1).indices

        # Get the visual concepts
        weighted_tokens = self.compute_weighted_tokens(visual_tokens, topk_indices)

        # Plot tmp data
        self.plot_tmp(images[0], topk_indices, visual_tokens, text_features, raw_text)

        # Get the corresponding text concepts
        textual_concepts = self.model.encode_text(
            self.concepts_tokens[topk_indices].reshape(
                self.args.batch_size * self.args.n_tokens, -1
            ),
            normalize=True,
        ).reshape(self.args.batch_size, self.args.n_tokens, -1)

        out_dict = {
            "text_features": text_features,
            "image_features": image_features,
            "visual_concepts": weighted_tokens,
            "textual_concepts": textual_concepts,
            "logit_scale": self.model.logit_scale.exp(),
            "topk_indices": topk_indices,
            "concepts_embeddings_normed": self.concepts_embeddings_normed,
        }
        if self.model.logit_bias is not None:
            out_dict["logit_bias"] = self.model.logit_bias

        return out_dict


class CLIPHead(nn.Module):
    def __init__(
        self,
        layer_norm,
        projection,
        all_classes_embeddings,
        all_classes_names,
        classes_embeddings,
        classes_names,
        num_heads=1,
        lambda_modality=0.1,
    ):
        super().__init__()
        self.layer_norm = layer_norm
        self.projection = projection
        self.words = all_classes_names
        self.class_names = classes_names

        # Set the last layer with normalized text embeddings
        self.all_classes_embeddings = all_classes_embeddings
        self.all_classes_embeddings_normed = F.normalize(
            all_classes_embeddings, p=2, dim=-1
        )

        # Store the class embeddings
        self.classes_embeddings_normed = F.normalize(classes_embeddings, p=2, dim=-1)

        # Attention layer
        self.attention_layer = MMAttention(
            dim=all_classes_embeddings.shape[-1],
            num_heads=num_heads,
        )
        self.lambda_modality = lambda_modality

    def forward(self, x):
        spatial_tokens = None
        if isinstance(x, tuple):
            x, spatial_tokens = x
            spatial_tokens = self.layer_norm(spatial_tokens)
            spatial_tokens = spatial_tokens @ self.projection

        # Project to same space as text
        x = self.layer_norm(x)
        x = x @ self.projection

        # Compute the predictions
        if spatial_tokens is not None:
            with torch.no_grad():
                class_predictions = spatial_tokens @ self.classes_embeddings_normed.T
        else:
            with torch.no_grad():
                class_predictions = x @ self.classes_embeddings_normed.T

        # Cross-modality attention (is all you need)
        queries = nn.functional.normalize(x, dim=-1, p=2)
        keys = self.all_classes_embeddings_normed
        values = self.all_classes_embeddings
        attended_x, _ = self.attention_layer(queries, keys, values)
        x = nn.functional.normalize(x, dim=-1, p=2)
        attended_x = nn.functional.normalize(attended_x, dim=-1, p=2)
        x = (1.0 - self.lambda_modality) * x + self.lambda_modality * attended_x

        # DINO head
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x, class_predictions


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
