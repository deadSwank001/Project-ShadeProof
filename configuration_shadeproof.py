"""Project-ShadeProof v2 Hugging Face configuration."""

from typing import Any, Dict


class ShadeProofConfig:
    model_type = "shadeproof"

    def __init__(
        self,
        surrogate_backbone: str = "openai/clip-vit-base-patch32",
        epsilon_budget: float = 8.0 / 255.0,
        defense_type: str = "adaptive_reversal",
        defense_strength: float = 1.0,
        anomaly_threshold: float = 0.15,
        embedding_dim: int = 512,
        image_size: int = 224,
        compatibility_mode: str = "v1",
        runtime_profile: str = "default",
        **kwargs: Any,
    ):
        legacy_defense = kwargs.pop("defense", None)
        if legacy_defense and not defense_type:
            defense_type = legacy_defense

        self.surrogate_backbone = surrogate_backbone
        self.epsilon_budget = epsilon_budget
        self.defense_type = defense_type
        self.defense_strength = defense_strength
        self.anomaly_threshold = anomaly_threshold
        self.embedding_dim = embedding_dim
        self.image_size = image_size
        self.compatibility_mode = compatibility_mode
        self.runtime_profile = runtime_profile
        self.architectures = ["ShadeProofSanitizer"]
        self.version = "2.0"

        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "architectures": self.architectures,
            "surrogate_backbone": self.surrogate_backbone,
            "epsilon_budget": self.epsilon_budget,
            "defense_type": self.defense_type,
            "defense": self.defense_type,
            "defense_strength": self.defense_strength,
            "anomaly_threshold": self.anomaly_threshold,
            "embedding_dim": self.embedding_dim,
            "image_size": self.image_size,
            "compatibility_mode": self.compatibility_mode,
            "runtime_profile": self.runtime_profile,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ShadeProofConfig":
        data = dict(config_dict)
        if "defense" in data and "defense_type" not in data:
            data["defense_type"] = data["defense"]
        return cls(**data)
