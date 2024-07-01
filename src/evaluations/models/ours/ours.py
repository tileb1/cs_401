from open_clip.model import OursCLIP
import open_clip
import torch
from .utils.prompt_templates import imagenet_templates
from typing import List, Optional
from models.builder import MODELS
from einops import rearrange
from torch import nn
import torch.nn.functional as F
import math

@MODELS.register_module()
class OursCLIPInference(OursCLIP):

    def __init__(
            self,
            class_names,
            visual_norm,
            model_name,
            use_templates,
            *args,
            **kwargs,
    ):
        super(OursCLIP, self).__init__(*args, **kwargs)
        self.use_templates = use_templates
        self.visual_norm = visual_norm
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.class_names = class_names
        self.handle_bkg = False

    def register_class_embeddings(self):
        self.register_buffer("class_embeddings", self._get_class_embeddings(self, self.class_names))

    @torch.no_grad()
    def _embed_label(self, text_model: torch.nn.Module, label: str) -> torch.Tensor:
        """
        Encode label name into a single vector
        """
        # Infer the device
        device = next(iter(text_model.parameters())).device

        if self.use_templates:
            templates = imagenet_templates
        elif "laion" in self.pretrained:
            templates = ['a photo of a {}', 'a photo of an {}']
        else:
            templates = ['a {}']
        all_prompts = [self.tokenizer(template.format(label)) for template in templates]
        out = text_model.encode_text(torch.cat(all_prompts).to(device))
        out /= out.norm(dim=-1, keepdim=True)
        out = out.mean(dim=0)
        return out

    def _get_class_embeddings(self, text_model: torch.nn.Module, class_names: List[str]):
        aug_embeddings = torch.stack([self._embed_label(text_model, label) for label in class_names])

        # Normalize vector
        aug_embeddings = aug_embeddings / aug_embeddings.norm(dim=-1, keepdim=True)
        return aug_embeddings.squeeze(1)
    
    def make_input_divisible(self, x: torch.Tensor) -> torch.Tensor:
        """Pad some pixels to make the input size divisible by the patch size."""
        _, _, h, w = x.shape
        pad_w = (self.visual.patch_size - w % self.visual.patch_size) % self.visual.patch_size
        pad_h = (self.visual.patch_size - h % self.visual.patch_size) % self.visual.patch_size
        x = nn.functional.pad(x, (0, pad_w, 0, pad_h), value=0)
        return x

    def forward(self, x):
        # Store the shape of the image
        b, _, h, w = x.shape

        # Pad
        x = self.make_input_divisible(x)

        # Forward the image
        x = self.encode_image(x, normalize=True)["x_patchtokens"]

        # Predict the class of the patches
        x = torch.einsum('b n d, c d -> b n c', x, self.class_embeddings)

        # Reshape to 2d
        h_patch = -(-h // self.visual.patch_size)
        w_patch = -(-w // self.visual.patch_size)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h_patch, w=w_patch)
        return x
