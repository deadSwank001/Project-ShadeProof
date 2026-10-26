from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter


class SanitizationDefensePipeline:
    @staticmethod
    def apply_defense(tensor_img: torch.Tensor, defense_type: str = "none", strength: float = 1.0) -> torch.Tensor:
        if defense_type == "none" or strength <= 0:
            return tensor_img.clone()

        if defense_type == "adaptive_reversal":
            return SanitizationDefensePipeline._adaptive_reversal(tensor_img, strength)

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
            kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32, device=device)
            kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
            high_freq = F.conv2d(tensor_img, kernel, padding=1, groups=3)
            sanitized = tensor_img - (0.1 * strength * high_freq)
            return torch.clamp(sanitized, 0.0, 1.0)

        purified_np = np.array(pil_img, dtype=np.float32) / 255.0
        purified_tensor = torch.from_numpy(purified_np).permute(2, 0, 1).unsqueeze(0).to(device)
        return purified_tensor

    @staticmethod
    def _adaptive_reversal(tensor_img: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        """Reverse likely NightShade-style high-frequency perturbations with conservative smoothing."""
        kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=tensor_img.dtype, device=tensor_img.device)
        kernel = (kernel / kernel.sum()).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        low = F.conv2d(tensor_img, kernel, padding=1, groups=3)

        lap = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=tensor_img.dtype, device=tensor_img.device)
        lap = lap.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        high = F.conv2d(tensor_img, lap, padding=1, groups=3)

        noise_gate = torch.sigmoid((torch.abs(high) - 0.08) * 12.0)
        repaired = low * noise_gate + tensor_img * (1.0 - noise_gate)
        correction = torch.clamp(repaired + (tensor_img - repaired) * (1.0 - min(1.0, strength / 2.0)), 0.0, 1.0)
        return correction

    @staticmethod
    def detect_poison_probability(raw: torch.Tensor, sanitized: torch.Tensor) -> torch.Tensor:
        delta = torch.mean(torch.abs(raw - sanitized), dim=(1, 2, 3))
        return torch.clamp((delta - 0.005) / 0.05, 0.0, 1.0)
