import math
import torch
import torch.nn.functional as F


class AcademicMetrics:
    @staticmethod
    def compute_linf(x_orig: torch.Tensor, x_adv: torch.Tensor) -> float:
        return float(torch.max(torch.abs(x_adv - x_orig)).item())

    @staticmethod
    def compute_l2(x_orig: torch.Tensor, x_adv: torch.Tensor) -> float:
        diff = (x_adv - x_orig).view(x_orig.size(0), -1)
        return float(torch.norm(diff, p=2, dim=1).mean().item())

    @staticmethod
    def compute_psnr(x_orig: torch.Tensor, x_adv: torch.Tensor, eps: float = 1e-10) -> float:
        mse = torch.mean((x_orig - x_adv) ** 2).item()
        if mse < eps:
            return 100.0
        return float(10.0 * math.log10(1.0 / mse))

    @staticmethod
    def compute_pseudo_ssim(x_orig: torch.Tensor, x_adv: torch.Tensor) -> float:
        c1 = (0.01) ** 2
        c2 = (0.03) ** 2
        mu_x = torch.mean(x_orig)
        mu_y = torch.mean(x_adv)
        sigma_x = torch.var(x_orig)
        sigma_y = torch.var(x_adv)
        sigma_xy = torch.mean((x_orig - mu_x) * (x_adv - mu_y))
        ssim = ((2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
        return float(torch.clamp(ssim, -1.0, 1.0).item())

    @staticmethod
    def cosine_similarity(z_a: torch.Tensor, z_b: torch.Tensor) -> float:
        z_a_norm = F.normalize(z_a, p=2, dim=-1)
        z_b_norm = F.normalize(z_b, p=2, dim=-1)
        return float(torch.sum(z_a_norm * z_b_norm).item())
