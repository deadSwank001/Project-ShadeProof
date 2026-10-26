from typing import Any, Dict

from .orchestrator import ExperimentOrchestrator


DEFAULT_PLUTO_PROFILE: Dict[str, Any] = {
    "defense_type": "adaptive_reversal",
    "defense_strength": 1.4,
    "iterations": 35,
    "epsilon": 6.0 / 255.0,
    "alpha": 1.5 / 255.0,
}


def run_pluto_profile(overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = dict(DEFAULT_PLUTO_PROFILE)
    cfg.update(overrides or {})
    return ExperimentOrchestrator(cfg).run()
