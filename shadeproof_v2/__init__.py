"""Project ShadeProof v2 core package."""

from .metrics import AcademicMetrics
from .defenses import SanitizationDefensePipeline
from .features import SurrogateFeatureModel
from .attacks import SurrogateAttackEngine
from .orchestrator import ExperimentOrchestrator, run_headless_benchmark
from .compat import normalize_config, normalize_results

__all__ = [
    "AcademicMetrics",
    "SanitizationDefensePipeline",
    "SurrogateFeatureModel",
    "SurrogateAttackEngine",
    "ExperimentOrchestrator",
    "run_headless_benchmark",
    "normalize_config",
    "normalize_results",
]
