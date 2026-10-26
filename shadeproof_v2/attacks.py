from typing import Tuple

import torch


class SurrogateAttackEngine:
    def __init__(self, surrogate, epsilon: float = 8.0 / 255.0, alpha: float = 2.0 / 255.0):
        self.surrogate = surrogate
        self.epsilon = epsilon
        self.alpha = alpha

    def pgd_step(self, x_orig: torch.Tensor, x_adv: torch.Tensor, z_target: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        x_adv_var = x_adv.clone().detach().requires_grad_(True)
        z_img = self.surrogate.encode_image(x_adv_var)
        cosine_sim = torch.sum(z_img * z_target)
        loss = -cosine_sim
        loss.backward()

        with torch.no_grad():
            grad_sign = x_adv_var.grad.sign()
            x_adv_next = x_adv_var - self.alpha * grad_sign
            delta = torch.clamp(x_adv_next - x_orig, -self.epsilon, self.epsilon)
            x_adv_clamped = torch.clamp(x_orig + delta, 0.0, 1.0)

        return x_adv_clamped.detach(), float(loss.item()), float(cosine_sim.item())
