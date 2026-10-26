import unittest

import torch

from shadeproof_v2.defenses import SanitizationDefensePipeline


class DefensePipelineTests(unittest.TestCase):
    def test_adaptive_reversal_shape(self):
        x = torch.rand(1, 3, 64, 64)
        y = SanitizationDefensePipeline.apply_defense(x, defense_type="adaptive_reversal", strength=1.2)
        self.assertEqual(tuple(x.shape), tuple(y.shape))

    def test_poison_probability_range(self):
        x = torch.rand(2, 3, 32, 32)
        y = SanitizationDefensePipeline.apply_defense(x, defense_type="gaussian_blur", strength=1.0)
        p = SanitizationDefensePipeline.detect_poison_probability(x, y)
        self.assertTrue(torch.all(p >= 0.0).item())
        self.assertTrue(torch.all(p <= 1.0).item())


if __name__ == "__main__":
    unittest.main()
