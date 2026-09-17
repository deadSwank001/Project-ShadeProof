"""
Project-ShadeProof: Scientific Backend & Adversarial Surrogate Benchmarking Suite
Copyright (c) 2026 David Burmeister. All rights reserved.
Licensed under the Apache License, Version 2.0.

Academic Threat Model & Evaluation Engine:
- Formulates surrogate-model adversarial perturbations targeting Vision-Language (CLIP) embeddings.
- Implements Projected Gradient Descent (PGD-L_inf) and FGSM optimization loops.
- Provides data sanitization and defense pipelines (JPEG, Spatial Filtering, Anomaly Scoring).
- Computes formal empirical metrics: L_inf, L_2, PSNR, SSIM, and Cosine Latent Shift.
- Dual-mode logging: Neptune Cloud Telemetry & Local Headless JSON/CSV artifact generation.
"""

import os
import time
import math
import json
from io import BytesIO
from typing import Dict, Any, Optional, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter

try:
    from PyQt6.QtCore import QThread, pyqtSignal
except ImportError:
    # Graceful fallback for headless research clusters and CLI execution without Qt
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


# ==============================================================================
# 1. ACADEMIC METRICS & MATHEMATICAL UTILITIES
# ==============================================================================

class AcademicMetrics:
    """
    Standardized empirical metrics for evaluating adversarial perturbations,
    perceptual degradation, and latent representation shifts.
    """

    @staticmethod
    def compute_linf(x_orig: torch.Tensor, x_adv: torch.Tensor) -> float:
        """Computes Chebyshev (L_infinity) maximum pixel distortion in [0, 1] range."""
        return float(torch.max(torch.abs(x_adv - x_orig)).item())

    @staticmethod
    def compute_l2(x_orig: torch.Tensor, x_adv: torch.Tensor) -> float:
        """Computes root mean square L2 distortion per pixel."""
        diff = (x_adv - x_orig).view(x_orig.size(0), -1)
        return float(torch.norm(diff, p=2, dim=1).mean().item())

    @staticmethod
    def compute_psnr(x_orig: torch.Tensor, x_adv: torch.Tensor, eps: float = 1e-10) -> float:
        """
        Peak Signal-to-Noise Ratio (PSNR) in dB:
        PSNR = 10 * log10(MAX_I^2 / MSE)
        Higher indicates greater visual fidelity (typically >30 dB is imperceptible).
        """
        mse = torch.mean((x_orig - x_adv) ** 2).item()
        if mse < eps:
            return 100.0
        return float(10.0 * math.log10(1.0 / mse))

    @staticmethod
    def compute_pseudo_ssim(x_orig: torch.Tensor, x_adv: torch.Tensor) -> float:
        """
        Mean Structural Similarity Index (SSIM) approximation across tensor channels.
        Ranges from -1 to 1 (1.0 = identical structure).
        """
        c1 = (0.01) ** 2
        c2 = (0.03) ** 2

        mu_x = torch.mean(x_orig)
        mu_y = torch.mean(x_adv)
        sigma_x = torch.var(x_orig)
        sigma_y = torch.var(x_adv)
        sigma_xy = torch.mean((x_orig - mu_x) * (x_adv - mu_y))

        ssim = ((2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)) / (
            (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
        )
        return float(torch.clamp(ssim, -1.0, 1.0).item())

    @staticmethod
    def cosine_similarity(z_a: torch.Tensor, z_b: torch.Tensor) -> float:
        """Cosine similarity between normalized feature representations in [-1, 1]."""
        z_a_norm = F.normalize(z_a, p=2, dim=-1)
        z_b_norm = F.normalize(z_b, p=2, dim=-1)
        return float(torch.sum(z_a_norm * z_b_norm).item())


# ==============================================================================
# 2. SANITIZATION & DEFENSE PIPELINES
# ==============================================================================

class SanitizationDefensePipeline:
    """
    Dataset sanitization transformations designed to mitigate surrogate adversarial
    poisoning before downstream generative model ingestion.
    """

    @staticmethod
    def apply_defense(
        tensor_img: torch.Tensor, 
        defense_type: str = "none", 
        strength: float = 1.0
    ) -> torch.Tensor:
        """
        Applies a defensive purification filter to an image tensor of shape [1, C, H, W] in [0, 1].
        Supported defenses: 'none', 'gaussian_blur', 'median_filter', 'jpeg_compression', 'total_variation'
        """
        if defense_type == "none" or strength <= 0:
            return tensor_img.clone()

        device = tensor_img.device
        img_np = tensor_img.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)

        if defense_type == "gaussian_blur":
            radius = max(0.5, strength * 1.5)
            pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))

        elif defense_type == "median_filter":
            size = 3 if strength <= 1.0 else 5
            pil_img = pil_img.filter(ImageFilter.MedianFilter(size=size))

        elif defense_type == "jpeg_compression":
            quality = max(20, min(95, int(100 - (strength * 25))))
            buffer = BytesIO()
            pil_img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            pil_img = Image.open(buffer)

        elif defense_type == "total_variation":
            # High-frequency gradient clipping
            kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32, device=device)
            kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
            high_freq = F.conv2d(tensor_img, kernel, padding=1, groups=3)
            sanitized = tensor_img - (0.1 * strength * high_freq)
            return torch.clamp(sanitized, 0.0, 1.0)

        # Reconvert PIL back to normalized tensor
        purified_np = np.array(pil_img, dtype=np.float32) / 255.0
        purified_tensor = torch.from_numpy(purified_np).permute(2, 0, 1).unsqueeze(0).to(device)
        return purified_tensor


# ==============================================================================
# 3. SURROGATE FEATURE EXTRACTOR (REAL CLIP / SYNTHETIC FALLBACK)
# ==============================================================================

class SurrogateFeatureModel:
    """
    Surrogate vision-language feature extractor.
    Attempts to load HuggingFace CLIP; gracefully falls back to a deterministic
    scientific surrogate model if offline or low-compute.
    """

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
            # Deterministic academic surrogate for offline / headless reproducibility
            self.is_real_clip = False
            self._init_fallback_surrogate()

    def _init_fallback_surrogate(self):
        """Constructs an analytical feature projection for gradient optimization verification."""
        torch.manual_seed(42)
        encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512),
        ).to(self.device)
        encoder.eval()
        self.model = encoder

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts normalized latent visual embedding z_img in R^512."""
        if self.is_real_clip:
            # CLIP expects normalized input tensors in standard range
            feats = self.model.get_image_features(pixel_values=x)
            return F.normalize(feats, p=2, dim=-1)
        else:
            feats = self.model(x)
            return F.normalize(feats, p=2, dim=-1)

    def encode_text(self, text: str) -> torch.Tensor:
        """Extracts normalized text concept embedding z_text in R^512."""
        if self.is_real_clip:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                feats = self.model.get_text_features(**inputs)
            return F.normalize(feats, p=2, dim=-1)
        else:
            # Deterministic pseudo-embedding based on text hash
            h = sum(ord(c) for c in text)
            torch.manual_seed(h)
            synthetic_text_emb = torch.randn((1, 512), device=self.device)
            return F.normalize(synthetic_text_emb, p=2, dim=-1)


# ==============================================================================
# 4. THREAT MODEL & PROJECTED GRADIENT DESCENT (PGD) ATTACK ENGINE
# ==============================================================================

class SurrogateAttackEngine:
    """
    Executes white-box surrogate adversarial optimization.
    Minimizes: L_adv(x') = - cos(f_img(x + delta), z_target)
    Subject to: ||delta||_inf <= epsilon, delta in [-epsilon, epsilon], x + delta in [0, 1]
    """

    def __init__(self, surrogate: SurrogateFeatureModel, epsilon: float = 8.0 / 255.0, alpha: float = 2.0 / 255.0):
        self.surrogate = surrogate
        self.epsilon = epsilon
        self.alpha = alpha

    def pgd_step(
        self, 
        x_orig: torch.Tensor, 
        x_adv: torch.Tensor, 
        z_target: torch.Tensor
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Executes a single Projected Gradient Descent iteration.
        Returns: (x_adv_next, loss, target_cosine)
        """
        x_adv_var = x_adv.clone().detach().requires_grad_(True)
        z_img = self.surrogate.encode_image(x_adv_var)

        # Loss is negative cosine similarity to target
        cosine_sim = torch.sum(z_img * z_target)
        loss = -cosine_sim

        loss.backward()

        with torch.no_grad():
            grad_sign = x_adv_var.grad.sign()
            # Gradient descent to maximize target cosine similarity
            x_adv_next = x_adv_var - self.alpha * grad_sign

            # Projection step 1: Clip to L_inf epsilon ball around original x
            delta = torch.clamp(x_adv_next - x_orig, -self.epsilon, self.epsilon)
            # Projection step 2: Clip to valid RGB image bounds [0, 1]
            x_adv_clamped = torch.clamp(x_orig + delta, 0.0, 1.0)

        return x_adv_clamped.detach(), float(loss.item()), float(cosine_sim.item())


# ==============================================================================
# 5. EXPERIMENT WORKER THREAD (PyQt6 & TELEMETRY)
# ==============================================================================

class ExperimentWorker(QThread):
    """
    Asynchronous QThread worker for executing rigorous adversarial evaluation loops,
    data sanitization tests, and empirical metrics logging.
    """

    # Signals
    step_progress = pyqtSignal(int, int, float, float, float, float)  # step, total, loss, tgt_cos, src_cos, l_inf
    status_updated = pyqtSignal(str)
    metric_logged = pyqtSignal(str, float)
    experiment_finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        config: Dict[str, Any],
        parent: Optional[Any] = None,
    ):
        super().__init__(parent)
        self.config = config
        self._is_running = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def stop(self):
        """Thread-safe cancellation handler."""
        self._is_running = False

    def run(self):
        run = None
        results_summary = {}
        try:
            total_steps = int(self.config.get("iterations", 50))
            epsilon = float(self.config.get("epsilon", 8.0 / 255.0))
            alpha = float(self.config.get("alpha", 2.0 / 255.0))
            source_prompt = self.config.get("source_prompt", "a photograph of a dog")
            target_prompt = self.config.get("target_prompt", "a luxury leather handbag")
            defense_type = self.config.get("defense_type", "none")
            defense_strength = float(self.config.get("defense_strength", 1.0))
            use_neptune = bool(self.config.get("use_neptune", False))
            neptune_project = self.config.get("neptune_project", "")
            neptune_api_token = self.config.get("neptune_api_token") or os.getenv("NEPTUNE_API_TOKEN")

            self.status_updated.emit(f"Initializing Experiment on {self.device.upper()}...")

            # Initialize Neptune if requested
            if use_neptune and neptune_project:
                try:
                    import neptune
                    run = neptune.init_run(
                        project=neptune_project,
                        api_token=neptune_api_token,
                        tags=["surrogate-evaluation", "shadeproof", defense_type, self.device],
                    )
                    run["parameters"] = {
                        "epsilon": epsilon,
                        "alpha": alpha,
                        "iterations": total_steps,
                        "source_prompt": source_prompt,
                        "target_prompt": target_prompt,
                        "defense_type": defense_type,
                        "defense_strength": defense_strength,
                        "device": self.device,
                    }
                    self.status_updated.emit("Neptune experiment initialized.")
                except Exception as neptune_err:
                    self.status_updated.emit(f"Neptune offline fallback: {neptune_err}")
                    run = None

            # 1. Load Surrogate Vision-Language Architecture
            surrogate_id = self.config.get("surrogate_model", "openai/clip-vit-base-patch32")
            self.status_updated.emit(f"Loading surrogate model '{surrogate_id}'...")
            surrogate = SurrogateFeatureModel(model_id=surrogate_id, device=self.device)
            mode_label = "HuggingFace CLIP" if surrogate.is_real_clip else "Analytical Surrogate (Offline)"
            self.status_updated.emit(f"Feature Extractor ready [{mode_label}].")

            # 2. Encode Semantic Prompts
            z_src = surrogate.encode_text(source_prompt)
            z_tgt = surrogate.encode_text(target_prompt)

            # 3. Create or Load Clean Image Tensor [1, 3, 224, 224]
            # Provide an authentic synthesized gradient-friendly natural pattern
            torch.manual_seed(101)
            H, W = 224, 224
            grid_y, grid_x = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij")
            clean_r = 0.5 + 0.3 * torch.sin(grid_x * 12.0) * torch.cos(grid_y * 12.0)
            clean_g = 0.5 + 0.3 * torch.cos(grid_x * 8.0)
            clean_b = 0.5 + 0.3 * torch.sin(grid_y * 8.0)
            x_clean = torch.stack([clean_r, clean_g, clean_b], dim=0).unsqueeze(0).to(self.device)
            x_clean = torch.clamp(x_clean, 0.0, 1.0)

            # Initial baseline metrics
            z_clean_img = surrogate.encode_image(x_clean)
            init_src_cos = AcademicMetrics.cosine_similarity(z_clean_img, z_src)
            init_tgt_cos = AcademicMetrics.cosine_similarity(z_clean_img, z_tgt)

            self.status_updated.emit(
                f"Baseline: Source Cosine = {init_src_cos:.4f}, Target Cosine = {init_tgt_cos:.4f}"
            )

            # 4. Initialize Attack Engine
            attack_engine = SurrogateAttackEngine(surrogate, epsilon=epsilon, alpha=alpha)
            x_adv = x_clean.clone()

            # Tracking histories
            history_steps = []
            history_loss = []
            history_tgt_cos = []
            history_src_cos = []
            history_linf = []

            # 5. Execute Adversarial Optimization Loop (PGD)
            start_time = time.time()
            for step in range(1, total_steps + 1):
                if not self._is_running:
                    self.status_updated.emit("Experiment canceled by user.")
                    break

                x_adv, loss, tgt_cos = attack_engine.pgd_step(x_clean, x_adv, z_tgt)

                # Compute current alignment with source concept
                with torch.no_grad():
                    z_curr = surrogate.encode_image(x_adv)
                    src_cos = AcademicMetrics.cosine_similarity(z_curr, z_src)
                    current_linf = AcademicMetrics.compute_linf(x_clean, x_adv)

                history_steps.append(step)
                history_loss.append(loss)
                history_tgt_cos.append(tgt_cos)
                history_src_cos.append(src_cos)
                history_linf.append(current_linf)

                # Telemetry logging
                if run:
                    run["metrics/adversarial_loss"].append(loss, step=step)
                    run["metrics/target_cosine_similarity"].append(tgt_cos, step=step)
                    run["metrics/source_cosine_similarity"].append(src_cos, step=step)
                    run["metrics/l_infinity_norm"].append(current_linf, step=step)

                # Emit step progress to GUI
                self.step_progress.emit(step, total_steps, loss, tgt_cos, src_cos, current_linf)
                self.metric_logged.emit("target_cosine", tgt_cos)

                # GUI responsive pacing
                self.msleep(20)

            elapsed = time.time() - start_time

            # 6. Apply Defensive Sanitization Layer
            self.status_updated.emit(f"Applying defense sanitization [{defense_type}]...")
            x_sanitized = SanitizationDefensePipeline.apply_defense(
                x_adv, defense_type=defense_type, strength=defense_strength
            )

            # 7. Post-Defense Academic Evaluation
            with torch.no_grad():
                z_sanitized = surrogate.encode_image(x_sanitized)
                post_tgt_cos = AcademicMetrics.cosine_similarity(z_sanitized, z_tgt)
                post_src_cos = AcademicMetrics.cosine_similarity(z_sanitized, z_src)

            final_linf = AcademicMetrics.compute_linf(x_clean, x_adv)
            final_l2 = AcademicMetrics.compute_l2(x_clean, x_adv)
            final_psnr = AcademicMetrics.compute_psnr(x_clean, x_adv)
            final_ssim = AcademicMetrics.compute_pseudo_ssim(x_clean, x_adv)

            # Sanitization metrics
            sanitization_delta_tgt = tgt_cos - post_tgt_cos
            sanitization_recovery_src = post_src_cos - src_cos
            anomaly_discrepancy = 1.0 - AcademicMetrics.cosine_similarity(z_curr, z_sanitized)

            results_summary = {
                "source_prompt": source_prompt,
                "target_prompt": target_prompt,
                "iterations": total_steps,
                "epsilon": epsilon,
                "alpha": alpha,
                "elapsed_seconds": elapsed,
                "surrogate_mode": mode_label,
                "defense_type": defense_type,
                "baseline_tgt_cos": init_tgt_cos,
                "baseline_src_cos": init_src_cos,
                "adversarial_final_tgt_cos": tgt_cos,
                "adversarial_final_src_cos": src_cos,
                "target_cosine_shift": tgt_cos - init_tgt_cos,
                "post_defense_tgt_cos": post_tgt_cos,
                "post_defense_src_cos": post_src_cos,
                "sanitization_target_drop": sanitization_delta_tgt,
                "sanitization_source_recovery": sanitization_recovery_src,
                "anomaly_score": anomaly_discrepancy,
                "l_infinity": final_linf,
                "l_2": final_l2,
                "psnr_db": final_psnr,
                "ssim": final_ssim,
                "history": {
                    "steps": history_steps,
                    "loss": history_loss,
                    "target_cosine": history_tgt_cos,
                    "source_cosine": history_src_cos,
                    "linf": history_linf,
                },
                # Numpy image representations [H, W, 3] in [0, 1] for GUI display
                "clean_np": x_clean.squeeze(0).permute(1, 2, 0).cpu().numpy(),
                "adv_np": x_adv.squeeze(0).permute(1, 2, 0).cpu().numpy(),
                "sanitized_np": x_sanitized.squeeze(0).permute(1, 2, 0).cpu().numpy(),
                "noise_np": torch.clamp(torch.abs(x_adv - x_clean) * 10.0, 0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy(),
            }

            # Final Neptune metadata
            if run:
                for k, v in results_summary.items():
                    if isinstance(v, (int, float, str, bool)):
                        run[f"summary/{k}"] = v

            self.status_updated.emit("Adversarial evaluation loop completed successfully.")
            self.experiment_finished.emit(results_summary)

        except Exception as exc:
            self.error_occurred.emit(str(exc))
        finally:
            if run:
                try:
                    run.stop()
                except Exception:
                    pass


# ==============================================================================
# 6. STANDALONE BENCHMARK UTILITY
# ==============================================================================

def run_headless_benchmark(
    iterations: int = 50,
    epsilon: float = 8.0 / 255.0,
    defense: str = "jpeg_compression",
    source_prompt: str = "a photograph of a dog",
    target_prompt: str = "a luxury leather handbag",
    output_dir: str = "./results",
) -> Dict[str, Any]:
    """
    Executes a complete headless benchmark run for command-line reproducibility
    and scientific ablation sweeps without requiring PyQt GUI event loops.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    surrogate = SurrogateFeatureModel(device=device)

    z_src = surrogate.encode_text(source_prompt)
    z_tgt = surrogate.encode_text(target_prompt)

    # Clean synthetic test pattern
    grid_y, grid_x = torch.meshgrid(torch.linspace(0, 1, 224), torch.linspace(0, 1, 224), indexing="ij")
    clean_tensor = torch.stack([
        0.5 + 0.3 * torch.sin(grid_x * 10.0),
        0.5 + 0.3 * torch.cos(grid_y * 10.0),
        0.5 + 0.2 * torch.sin((grid_x + grid_y) * 8.0)
    ], dim=0).unsqueeze(0).to(device)

    attack_engine = SurrogateAttackEngine(surrogate, epsilon=epsilon, alpha=2.0 / 255.0)
    x_adv = clean_tensor.clone()

    init_cos = AcademicMetrics.cosine_similarity(surrogate.encode_image(clean_tensor), z_tgt)
    for _ in range(iterations):
        x_adv, _, _ = attack_engine.pgd_step(clean_tensor, x_adv, z_tgt)

    x_sanitized = SanitizationDefensePipeline.apply_defense(x_adv, defense_type=defense, strength=1.0)

    final_adv_cos = AcademicMetrics.cosine_similarity(surrogate.encode_image(x_adv), z_tgt)
    final_def_cos = AcademicMetrics.cosine_similarity(surrogate.encode_image(x_sanitized), z_tgt)

    metrics = {
        "device": device,
        "iterations": iterations,
        "epsilon": epsilon,
        "defense_type": defense,
        "baseline_tgt_cos": init_cos,
        "adversarial_tgt_cos": final_adv_cos,
        "cosine_shift": final_adv_cos - init_cos,
        "sanitized_tgt_cos": final_def_cos,
        "psnr_db": AcademicMetrics.compute_psnr(clean_tensor, x_adv),
        "linf": AcademicMetrics.compute_linf(clean_tensor, x_adv),
    }

    timestamp = int(time.time())
    out_path = os.path.join(output_dir, f"benchmark_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
