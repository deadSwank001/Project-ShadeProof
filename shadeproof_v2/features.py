import torch
import torch.nn as nn
import torch.nn.functional as F


class SurrogateFeatureModel:
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        self.is_real_clip = False
        self.model = None
        self.processor = None
        self._init_model()

    def _init_model(self):
        try:
            from transformers import CLIPModel, CLIPProcessor
            self.model = CLIPModel.from_pretrained(self.model_id).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_id)
            self.model.eval()
            self.is_real_clip = True
        except Exception:
            self.is_real_clip = False
            self._init_fallback_surrogate()

    def _init_fallback_surrogate(self):
        torch.manual_seed(42)
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512),
        ).to(self.device)
        self.model.eval()

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_real_clip:
            feats = self.model.get_image_features(pixel_values=x)
        else:
            feats = self.model(x)
        return F.normalize(feats, p=2, dim=-1)

    def encode_text(self, text: str) -> torch.Tensor:
        if self.is_real_clip:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                feats = self.model.get_text_features(**inputs)
            return F.normalize(feats, p=2, dim=-1)

        h = sum(ord(c) for c in text)
        torch.manual_seed(h)
        synthetic = torch.randn((1, 512), device=self.device)
        return F.normalize(synthetic, p=2, dim=-1)
