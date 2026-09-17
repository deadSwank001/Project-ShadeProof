"""
Project-ShadeProof: Hugging Face Model Configuration
Copyright (c) 2026 David Burmeister. All rights reserved.
Licensed under the Apache License, Version 2.0.

Defines the PretrainedConfig for publishing ShadeProof as a Hugging Face model/macro.
"""

from typing import Dict, Any


class ShadeProofConfig:
    """
    Configuration class for the ShadeProof Sanitizer and Adversarial Defense model.
    Compatible with Hugging Face PretrainedConfig schemas.
    """
    model_type = "shadeproof"

    def __init__(
        self,
        surrogate_backbone: str = "openai/clip-vit-base-patch32",
        epsilon_budget: float = 8.0 / 255.0,
        defense_type: str = "jpeg_compression",
        defense_strength: float = 1.0,
        anomaly_threshold: float = 0.15,
        embedding_dim: int = 512,
        image_size: int = 224,
        **kwargs: Any,
    ):
        self.surrogate_backbone = surrogate_backbone
        self.epsilon_budget = epsilon_budget
        self.defense_type = defense_type
        self.defense_strength = defense_strength
        self.anomaly_threshold = anomaly_threshold
        self.embedding_dim = embedding_dim
        self.image_size = image_size
        self.architectures = ["ShadeProofSanitizer"]

        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes configuration to dictionary format."""
        return {
            "model_type": self.model_type,
            "architectures": self.architectures,
            "surrogate_backbone": self.surrogate_backbone,
            "epsilon_budget": self.epsilon_budget,
            "defense_type": self.defense_type,
            "defense_strength": self.defense_strength,
            "anomaly_threshold": self.anomaly_threshold,
            "embedding_dim": self.embedding_dim,
            "image_size": self.image_size,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ShadeProofConfig":
        """Instantiates configuration from dictionary."""
        return cls(**config_dict)
