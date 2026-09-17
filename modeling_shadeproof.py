"""
Project-ShadeProof: Hugging Face Modeling Architecture & Defense Macro
Copyright (c) 2026 David Burmeister. All rights reserved.
Licensed under the Apache License, Version 2.0.

Provides the ShadeProofSanitizer model architecture for direct Hugging Face Hub publication.
Can be loaded via:
    model = ShadeProofSanitizer.from_pretrained("deadSwank001/Project-ShadeProof")
"""

import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from configuration_shadeproof import ShadeProofConfig
from backend import SanitizationDefensePipeline, AcademicMetrics, SurrogateFeatureModel


@dataclass
class ShadeProofOutput:
    """Output structure returned by ShadeProofSanitizer forward passes."""
    sanitized_pixel_values: torch.Tensor
    anomaly_score: torch.Tensor
    is_poisoned: torch.Tensor
    residual_noise: torch.Tensor
    latent_embedding: Optional[torch.Tensor] = None


class ShadeProofSanitizer(nn.Module):
    """
    ShadeProof Defensive Macro Model for Text-to-Image Data Sanitization.
    Detects surrogate adversarial perturbations (e.g. Nightshade/Glaze concepts)
    and purifies image tensors prior to training ingestion.
    """
    config_class = ShadeProofConfig

    def __init__(self, config: Optional[ShadeProofConfig] = None):
        super().__init__()
        self.config = config or ShadeProofConfig()

        # Feature projection backbone for representation anomaly detection
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
        """Extracts normalized latent representation vectors in S^(d-1)."""
        feats = self.encoder(pixel_values)
        return F.normalize(feats, p=2, dim=-1)

    def sanitize(
        self, 
        pixel_values: torch.Tensor, 
        defense_type: Optional[str] = None, 
        strength: Optional[float] = None
    ) -> torch.Tensor:
        """Applies dataset sanitization filter to input batch."""
        def_type = defense_type or self.config.defense_type
        str_val = strength if strength is not None else self.config.defense_strength

        # Batch-wise defense application
        sanitized_list = []
        for i in range(pixel_values.size(0)):
            single_img = pixel_values[i:i+1]
            sanitized_single = SanitizationDefensePipeline.apply_defense(
                single_img, defense_type=def_type, strength=str_val
            )
            sanitized_list.append(sanitized_single)
        return torch.cat(sanitized_list, dim=0)

    def forward(
        self,
        pixel_values: torch.Tensor,
        defense_type: Optional[str] = None,
        defense_strength: Optional[float] = None,
        return_dict: bool = True,
    ) -> ShadeProofOutput:
        """
        Forward pass:
        1. Encodes raw visual representations: z_raw = f(x).
        2. Applies defensive sanitization operator: x_san = T_san(x).
        3. Encodes purified visual representations: z_san = f(x_san).
        4. Calculates representation discrepancy anomaly: S_anomaly = 1 - cos(z_raw, z_san).
        5. Computes poison flag based on anomaly threshold.
        """
        device = pixel_values.device
        self.to(device)

        # 1. Raw representation
        z_raw = self.extract_features(pixel_values)

        # 2. Sanitization
        sanitized_pixels = self.sanitize(
            pixel_values, 
            defense_type=defense_type, 
            strength=defense_strength
        ).to(device)

        # 3. Purified representation
        z_san = self.extract_features(sanitized_pixels)

        # 4. Latent representation discrepancy
        cosine_sim = torch.sum(z_raw * z_san, dim=-1)
        anomaly_score = torch.clamp(1.0 - cosine_sim, 0.0, 2.0)

        # 5. Poison classification
        is_poisoned = anomaly_score > self.config.anomaly_threshold

        # 6. Residual perturbation map
        residual_noise = torch.abs(pixel_values - sanitized_pixels)

        return ShadeProofOutput(
            sanitized_pixel_values=sanitized_pixels,
            anomaly_score=anomaly_score,
            is_poisoned=is_poisoned,
            residual_noise=residual_noise,
            latent_embedding=z_raw,
        )

    def save_pretrained(self, save_directory: str):
        """Saves model weights and configuration in standard Hugging Face format."""
        os.makedirs(save_directory, exist_ok=True)

        # Save config.json
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save pytorch_model.bin
        weights_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), weights_path)

        print(f"[+] Model checkpoint and config saved to: {save_directory}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str) -> "ShadeProofSanitizer":
        """Loads model weights and configuration from a local directory or checkpoint."""
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        weights_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")

        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            config = ShadeProofConfig.from_dict(config_dict)
        else:
            config = ShadeProofConfig()

        model = cls(config=config)
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)

        model.eval()
        return model
