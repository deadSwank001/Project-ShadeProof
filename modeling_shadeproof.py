"""Project-ShadeProof v2 sanitizer model."""

import json
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from configuration_shadeproof import ShadeProofConfig
from shadeproof_v2.defenses import SanitizationDefensePipeline


@dataclass
class ShadeProofOutput:
    sanitized_pixel_values: torch.Tensor
    anomaly_score: torch.Tensor
    is_poisoned: torch.Tensor
    residual_noise: torch.Tensor
    latent_embedding: Optional[torch.Tensor] = None
    confidence_score: Optional[torch.Tensor] = None


class ShadeProofSanitizer(nn.Module):
    config_class = ShadeProofConfig

    def __init__(self, config: Optional[ShadeProofConfig] = None):
        super().__init__()
        self.config = config or ShadeProofConfig()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, self.config.embedding_dim),
        )

    def extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encoder(pixel_values), p=2, dim=-1)

    def sanitize(self, pixel_values: torch.Tensor, defense_type: Optional[str] = None, strength: Optional[float] = None) -> torch.Tensor:
        def_type = defense_type or self.config.defense_type
        str_val = strength if strength is not None else self.config.defense_strength
        sanitized_list = []
        for i in range(pixel_values.size(0)):
            sanitized = SanitizationDefensePipeline.apply_defense(pixel_values[i : i + 1], defense_type=def_type, strength=str_val)
            sanitized_list.append(sanitized)
        return torch.cat(sanitized_list, dim=0)

    def forward(
        self,
        pixel_values: torch.Tensor,
        defense_type: Optional[str] = None,
        defense_strength: Optional[float] = None,
        return_dict: bool = True,
    ) -> ShadeProofOutput:
        z_raw = self.extract_features(pixel_values)
        sanitized = self.sanitize(pixel_values, defense_type=defense_type, strength=defense_strength).to(pixel_values.device)
        z_san = self.extract_features(sanitized)

        cosine_sim = torch.sum(z_raw * z_san, dim=-1)
        anomaly = torch.clamp(1.0 - cosine_sim, 0.0, 2.0)
        is_poisoned = anomaly > self.config.anomaly_threshold
        residual = torch.abs(pixel_values - sanitized)
        poison_prob = SanitizationDefensePipeline.detect_poison_probability(pixel_values, sanitized)
        confidence = 1.0 - poison_prob

        return ShadeProofOutput(
            sanitized_pixel_values=sanitized,
            anomaly_score=anomaly,
            is_poisoned=is_poisoned,
            residual_noise=residual,
            latent_embedding=z_raw,
            confidence_score=confidence,
        )

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str) -> "ShadeProofSanitizer":
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        weights_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")

        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = ShadeProofConfig.from_dict(json.load(f))
        else:
            config = ShadeProofConfig()

        model = cls(config=config)
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()
        return model
