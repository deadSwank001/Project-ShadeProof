"""
Project-ShadeProof backend API (v2).
Backwards-compatible public surface for GUI and headless workflows.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

try:
    from PyQt6.QtCore import QThread, pyqtSignal
except ImportError:
    import time

    class QThread:
        def __init__(self, parent=None):
            self.parent = parent

        def start(self):
            self.run()

        def msleep(self, ms):
            time.sleep(ms / 1000.0)

    class DummySignal:
        def emit(self, *args, **kwargs):
            pass

        def connect(self, fn):
            pass

    def pyqtSignal(*args, **kwargs):
        return DummySignal()

from shadeproof_v2 import (
    AcademicMetrics,
    SanitizationDefensePipeline,
    SurrogateAttackEngine,
    SurrogateFeatureModel,
    ExperimentOrchestrator,
    normalize_config,
    normalize_results,
    run_headless_benchmark,
)


class ExperimentWorker(QThread):
    step_progress = pyqtSignal(int, int, float, float, float, float)
    status_updated = pyqtSignal(str)
    metric_logged = pyqtSignal(str, float)
    experiment_finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, config: Dict[str, Any], parent: Optional[Any] = None):
        super().__init__(parent)
        self.config = normalize_config(config)
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            orchestrator = ExperimentOrchestrator(self.config)

            def _progress(step: int, total: int, loss: float, tgt_cos: float, src_cos: float, l_inf: float):
                self.step_progress.emit(step, total, loss, tgt_cos, src_cos, l_inf)
                self.metric_logged.emit("target_cosine", tgt_cos)
                self.msleep(20)

            def _status(msg: str):
                self.status_updated.emit(msg)

            results = orchestrator.run(
                progress_cb=_progress,
                status_cb=_status,
                stop_requested=lambda: not self._is_running,
            )
            self.status_updated.emit("Adversarial evaluation loop completed successfully.")
            self.experiment_finished.emit(normalize_results(results))
        except Exception as exc:
            self.error_occurred.emit(str(exc))


__all__ = [
    "AcademicMetrics",
    "SanitizationDefensePipeline",
    "SurrogateFeatureModel",
    "SurrogateAttackEngine",
    "ExperimentOrchestrator",
    "ExperimentWorker",
    "normalize_config",
    "normalize_results",
    "run_headless_benchmark",
]
