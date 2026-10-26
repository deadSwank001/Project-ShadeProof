import json
import os
import time
from typing import Any, Callable, Dict, Optional

import torch

from .attacks import SurrogateAttackEngine
from .compat import normalize_config, normalize_results
from .defenses import SanitizationDefensePipeline
from .features import SurrogateFeatureModel
from .metrics import AcademicMetrics


class ExperimentOrchestrator:
    """v2 orchestration with defense-first defaults and backwards-compatible outputs."""

    def __init__(self, raw_config: Optional[Dict[str, Any]] = None):
        self.config = normalize_config(raw_config or {})
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _build_clean_tensor(self) -> torch.Tensor:
        torch.manual_seed(101)
        h, w = 224, 224
        grid_y, grid_x = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
        clean_r = 0.5 + 0.3 * torch.sin(grid_x * 12.0) * torch.cos(grid_y * 12.0)
        clean_g = 0.5 + 0.3 * torch.cos(grid_x * 8.0)
        clean_b = 0.5 + 0.3 * torch.sin(grid_y * 8.0)
        return torch.clamp(torch.stack([clean_r, clean_g, clean_b], dim=0).unsqueeze(0).to(self.device), 0.0, 1.0)

    def run(
        self,
        progress_cb: Optional[Callable[[int, int, float, float, float, float], None]] = None,
        status_cb: Optional[Callable[[str], None]] = None,
        stop_requested: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        cfg = self.config
        total_steps = int(cfg["iterations"])
        epsilon = float(cfg["epsilon"])
        alpha = float(cfg["alpha"])

        if status_cb:
            status_cb(f"Initializing Experiment on {self.device.upper()}...")

        surrogate = SurrogateFeatureModel(model_id=cfg.get("surrogate_model", "openai/clip-vit-base-patch32"), device=self.device)
        mode_label = "HuggingFace CLIP" if surrogate.is_real_clip else "Analytical Surrogate (Offline)"

        z_src = surrogate.encode_text(cfg["source_prompt"])
        z_tgt = surrogate.encode_text(cfg["target_prompt"])

        x_clean = self._build_clean_tensor()
        x_adv = x_clean.clone()
        z_clean_img = surrogate.encode_image(x_clean)
        init_src_cos = AcademicMetrics.cosine_similarity(z_clean_img, z_src)
        init_tgt_cos = AcademicMetrics.cosine_similarity(z_clean_img, z_tgt)

        attack_engine = SurrogateAttackEngine(surrogate, epsilon=epsilon, alpha=alpha)

        history_steps, history_loss, history_tgt_cos, history_src_cos, history_linf = [], [], [], [], []

        start = time.time()
        tgt_cos, src_cos = init_tgt_cos, init_src_cos
        for step in range(1, total_steps + 1):
            if stop_requested and stop_requested():
                break

            x_adv, loss, tgt_cos = attack_engine.pgd_step(x_clean, x_adv, z_tgt)
            with torch.no_grad():
                z_curr = surrogate.encode_image(x_adv)
                src_cos = AcademicMetrics.cosine_similarity(z_curr, z_src)
                current_linf = AcademicMetrics.compute_linf(x_clean, x_adv)

            history_steps.append(step)
            history_loss.append(loss)
            history_tgt_cos.append(tgt_cos)
            history_src_cos.append(src_cos)
            history_linf.append(current_linf)

            if progress_cb:
                progress_cb(step, total_steps, loss, tgt_cos, src_cos, current_linf)

        elapsed = time.time() - start

        defense_type = cfg.get("defense_type", "adaptive_reversal")
        defense_strength = float(cfg.get("defense_strength", 1.0))
        if status_cb:
            status_cb(f"Applying defense sanitization [{defense_type}]...")

        x_sanitized = SanitizationDefensePipeline.apply_defense(x_adv, defense_type=defense_type, strength=defense_strength)
        poison_probability = float(SanitizationDefensePipeline.detect_poison_probability(x_adv, x_sanitized).mean().item())

        with torch.no_grad():
            z_sanitized = surrogate.encode_image(x_sanitized)
            post_tgt_cos = AcademicMetrics.cosine_similarity(z_sanitized, z_tgt)
            post_src_cos = AcademicMetrics.cosine_similarity(z_sanitized, z_src)

        final_linf = AcademicMetrics.compute_linf(x_clean, x_adv)
        results_summary = {
            "source_prompt": cfg["source_prompt"],
            "target_prompt": cfg["target_prompt"],
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
            "sanitization_target_drop": tgt_cos - post_tgt_cos,
            "sanitization_source_recovery": post_src_cos - src_cos,
            "anomaly_score": 1.0 - AcademicMetrics.cosine_similarity(surrogate.encode_image(x_adv), z_sanitized),
            "poison_probability": poison_probability,
            "l_infinity": final_linf,
            "l_2": AcademicMetrics.compute_l2(x_clean, x_adv),
            "psnr_db": AcademicMetrics.compute_psnr(x_clean, x_adv),
            "ssim": AcademicMetrics.compute_pseudo_ssim(x_clean, x_adv),
            "history": {
                "steps": history_steps,
                "loss": history_loss,
                "target_cosine": history_tgt_cos,
                "source_cosine": history_src_cos,
                "linf": history_linf,
            },
            "clean_np": x_clean.squeeze(0).permute(1, 2, 0).cpu().numpy(),
            "adv_np": x_adv.squeeze(0).permute(1, 2, 0).cpu().numpy(),
            "sanitized_np": x_sanitized.squeeze(0).permute(1, 2, 0).cpu().numpy(),
            "noise_np": torch.clamp(torch.abs(x_adv - x_clean) * 10.0, 0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy(),
        }
        return normalize_results(results_summary)


def run_headless_benchmark(
    iterations: int = 50,
    epsilon: float = 8.0 / 255.0,
    defense: str = "adaptive_reversal",
    source_prompt: str = "a photograph of a dog",
    target_prompt: str = "a luxury leather handbag",
    output_dir: str = "./results",
) -> Dict[str, Any]:
    cfg = {
        "iterations": iterations,
        "epsilon": epsilon,
        "defense_type": defense,
        "source_prompt": source_prompt,
        "target_prompt": target_prompt,
    }
    orchestrator = ExperimentOrchestrator(cfg)
    metrics = orchestrator.run()

    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    out_path = os.path.join(output_dir, f"benchmark_{timestamp}.json")

    serializable = {
        k: v for k, v in metrics.items() if isinstance(v, (int, float, str, bool, dict, list))
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    return metrics
