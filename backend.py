#Project-ShadeProof
#
# Copyright (c) 2026 David Burmeister. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import os
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F
from PyQt6.QtCore import QThread, pyqtSignal
from transformers import CLIPModel, CLIPProcessor
from diffusers import AutoencoderKL
import neptune


class ExperimentWorker(QThread):
    """
    Non-blocking worker thread for evaluating adversarial perturbations
    against surrogate vision-language and diffusion models.
    """
    progress = pyqtSignal(int, int)          # current_step, total_steps
    metric_logged = pyqtSignal(str, float)   # metric_name, value
    status_updated = pyqtSignal(str)        # status message
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        config: Dict[str, Any],
        surrogate_model_id: str = "openai/clip-vit-base-patch32",
        vae_model_id: str = "stabilityai/sd-vae-ft-mse",
        parent: Optional[Any] = None,
    ):
        super().__init__(parent)
        self.config = config
        self.surrogate_model_id = surrogate_model_id
        self.vae_model_id = vae_model_id
        self._is_running = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run(self):
        run = None
        try:
            self.status_updated.emit("Initializing Neptune experiment run...")
            run = neptune.init_run(
                project=self.config.get("neptune_project"),
                api_token=self.config.get("neptune_api_token") or os.getenv("NEPTUNE_API_TOKEN"),
                tags=["surrogate-analysis", "adversarial-eval", self.device],
            )
            
            # Log base parameters
            run["parameters"] = {
                "surrogate_model": self.surrogate_model_id,
                "vae_model": self.vae_model_id,
                "device": self.device,
                "iterations": self.config.get("iterations", 100),
                "learning_rate": self.config.get("lr", 1e-3),
                "epsilon": self.config.get("epsilon", 8 / 255),
            }

            self.status_updated.emit(f"Loading surrogate models onto {self.device}...")
            clip_model = CLIPModel.from_pretrained(self.surrogate_model_id).to(self.device)
            clip_processor = CLIPProcessor.from_pretrained(self.surrogate_model_id)
            vae = AutoencoderKL.from_pretrained(self.vae_model_id).to(self.device)

            clip_model.eval()
            vae.eval()

            # Synthetic batch allocation for pipeline demonstration
            total_steps = self.config.get("iterations", 100)
            target_text = self.config.get("target_prompt", "a generic handbag")
            
            text_inputs = clip_processor(
                text=[target_text], return_tensors="pt", padding=True
            ).to(self.device)
            with torch.no_grad():
                target_text_embedding = clip_model.get_text_features(**text_inputs)
                target_text_embedding = F.normalize(target_text_embedding, p=2, dim=-1)

            # Simulated optimization / evaluation loop
            for step in range(1, total_steps + 1):
                if not self._is_running:
                    self.status_updated.emit("Operation canceled by user.")
                    break

                # Dummy measurement tracking: cosine distance in latent embedding space
                # Replace with batch tensor evaluations during live sweeps
                synthetic_loss = (1.0 / step) + 0.05
                simulated_cosine_sim = 1.0 - synthetic_loss

                # Log directly to Neptune
                run["metrics/loss"].append(synthetic_loss, step=step)
                run["metrics/cosine_similarity"].append(simulated_cosine_sim, step=step)

                # Emit updates to UI
                self.progress.emit(step, total_steps)
                self.metric_logged.emit("cosine_similarity", simulated_cosine_sim)
                self.msleep(50)  # Rate pacing for GUI responsiveness

            self.status_updated.emit("Evaluation run finished successfully.")
            self.finished.emit()

        except Exception as exc:
            self.error_occurred.emit(str(exc))
        finally:
            if run:
                run.stop()

    def stop(self):
        """Thread-safe cancellation flag."""
        self._is_running = False
