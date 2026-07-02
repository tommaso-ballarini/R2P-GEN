"""
pipeline/metrics_dreambench.py

Metrics used in the DreamBench paper (CLIP-I, CLIP-T, DINO-I) for the official benchmark table.

We replicate the original DreamBench protocol exactly, using the official checkpoints:
    - CLIP-I / CLIP-T: openai/clip-vit-base-patch32
    - DINO:            facebook/dino-vits16
calculated as direct image-image (CLIP-I, DINO) or image-text (CLIP-T) similarity, averaged pairwise 
over ALL real images of the subject (not just a single reference), as per the original paper.
"""

import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Union

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Config


def _load_image(img: Union[str, Image.Image]) -> Image.Image:
    if isinstance(img, str):
        return Image.open(img).convert("RGB")
    return img.convert("RGB")


class ClipDreamBench:
    """CLIP-I and CLIP-T with the official ViT-B/32 checkpoint (ImageNet-style
    direct cosine similarity, NO attribute-score, NO accept/reject logic — just numbers to report in the table)."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        model_id = getattr(Config.Models, "CLIP_DREAMBENCH_MODEL", "openai/clip-vit-base-patch32")
        print(f"   📦 Loading CLIP-DreamBench: {model_id}")
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(model_id, local_files_only=True).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)

    @torch.no_grad()
    def _image_features(self, img: Union[str, Image.Image]) -> torch.Tensor:
        img = _load_image(img)
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        out = self.model.vision_model(**inputs)
        feats = self.model.visual_projection(out.pooler_output)  # 768 → 512
        return F.normalize(feats, p=2, dim=-1)

    @torch.no_grad()
    def _text_features(self, text: str) -> torch.Tensor:
        inputs = self.processor(text=text, return_tensors="pt", truncation=True).to(self.device)
        out = self.model.text_model(**inputs)
        feats = self.model.text_projection(out.pooler_output)    # 512 → 512
        return F.normalize(feats, p=2, dim=-1)

    def clip_i(self, gen_image: Union[str, Image.Image], real_images: List[str]) -> float:
        """Pairwise mean between the generated image and ALL real images
        of the subject (official protocol, not just a single reference)."""
        gen_feat = self._image_features(gen_image)
        sims = []
        for real_path in real_images:
            real_feat = self._image_features(real_path)
            sims.append((gen_feat @ real_feat.T).item())
        return sum(sims) / len(sims) if sims else 0.0

    def clip_t(self, gen_image: Union[str, Image.Image], prompt: str) -> float:
        gen_feat = self._image_features(gen_image)
        text_feat = self._text_features(prompt)
        return (gen_feat @ text_feat.T).item()

    def cleanup(self):
        del self.model, self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class DinoDreamBench:
    """DINO-I with the original checkpoint from the DreamBooth paper: DINOv1 ViT-S/16
    (facebook/dino-vits16, Caron et al. 2021). The same checkpoint used in
    BLIP-Diffusion, ComFusion, DreamMatcher and in the vast majority
    of subject-driven generation literature.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        model_id = getattr(Config.Models, "DINO_DREAMBENCH_MODEL", "facebook/dino-vits16")
        print(f"   📦 Loading DINO-DreamBench: {model_id}")
        from transformers import ViTModel, ViTImageProcessor
        self.model = ViTModel.from_pretrained(
            model_id,
            add_pooling_layer=False,
            local_files_only=True,
        ).to(device).eval()
        self.processor = ViTImageProcessor.from_pretrained(   # <-- mancava
            model_id,
            local_files_only=True,
        )

    @torch.no_grad()
    def _features(self, img):
        img = _load_image(img)
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        out = self.model(**inputs, output_hidden_states=True)
        cls = out.hidden_states[-1][:, 0, :]
        return F.normalize(cls, p=2, dim=-1)

    def dino_i(self, gen_image: Union[str, Image.Image], real_images: List[str]) -> float:
        gen_feat = self._features(gen_image)
        sims = []
        for real_path in real_images:
            real_feat = self._features(real_path)
            sims.append((gen_feat @ real_feat.T).item())
        return sum(sims) / len(sims) if sims else 0.0

    def cleanup(self):
        del self.model, self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()