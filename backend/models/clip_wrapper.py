# backend/models/clip_wrapper.py
import clip
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from backend.config import CLIP_MODEL
from backend.core.device import get_device

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


class CLIPWrapper:
    def __init__(self, model_name: str = CLIP_MODEL):
        self.device = get_device()
        self.model, self._preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        self._mean = CLIP_MEAN.to(self.device)
        self._std = CLIP_STD.to(self.device)

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self._mean) / self._std

    def encode_image_differentiable(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[-2:] != (224, 224):
            tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
        normed = self._normalize(tensor)
        return self.model.encode_image(normed.to(self.model.dtype))

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        tokens = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            return self.model.encode_text(tokens).float()

    def classify(self, image: np.ndarray, labels: list[str]) -> dict[str, float]:
        pil_img = Image.fromarray(image)
        img_tensor = self._preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_features = self.model.encode_image(img_tensor).float()
            text_features = self.encode_text(labels)
            img_features = F.normalize(img_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            similarity = (img_features @ text_features.T).squeeze(0)
            probs = F.softmax(similarity * 100, dim=0)
        return {label: float(probs[i]) for i, label in enumerate(labels)}
