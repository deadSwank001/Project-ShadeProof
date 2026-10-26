from typing import Any, Dict

LEGACY_DEFENSE_MAP = {
    "tv": "total_variation",
    "jpeg": "jpeg_compression",
}


def normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(config)

    if "defense" in normalized and "defense_type" not in normalized:
        normalized["defense_type"] = normalized["defense"]
    if "eps" in normalized and "epsilon" not in normalized:
        normalized["epsilon"] = normalized["eps"]
    if "steps" in normalized and "iterations" not in normalized:
        normalized["iterations"] = normalized["steps"]
    if "target" in normalized and "target_prompt" not in normalized:
        normalized["target_prompt"] = normalized["target"]
    if "source" in normalized and "source_prompt" not in normalized:
        normalized["source_prompt"] = normalized["source"]

    defense_type = normalized.get("defense_type", "adaptive_reversal")
    normalized["defense_type"] = LEGACY_DEFENSE_MAP.get(defense_type, defense_type)

    normalized.setdefault("defense_type", "adaptive_reversal")
    normalized.setdefault("defense_strength", 1.0)
    normalized.setdefault("epsilon", 8.0 / 255.0)
    normalized.setdefault("alpha", 2.0 / 255.0)
    normalized.setdefault("iterations", 50)
    normalized.setdefault("source_prompt", "a photograph of a dog")
    normalized.setdefault("target_prompt", "a luxury leather handbag")

    return normalized


def normalize_results(results: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(results)
    out.setdefault("adversarial_tgt_cos", out.get("adversarial_final_tgt_cos", 0.0))
    out.setdefault("linf", out.get("l_infinity", 0.0))
    out.setdefault("defense", out.get("defense_type", "none"))
    out.setdefault("poison_probability", out.get("poison_probability", 0.0))
    return out
