import unittest

from shadeproof_v2.compat import normalize_config, normalize_results


class CompatibilityTests(unittest.TestCase):
    def test_legacy_config_keys_map(self):
        cfg = normalize_config({"eps": 0.02, "steps": 12, "source": "s", "target": "t", "defense": "jpeg"})
        self.assertEqual(cfg["epsilon"], 0.02)
        self.assertEqual(cfg["iterations"], 12)
        self.assertEqual(cfg["source_prompt"], "s")
        self.assertEqual(cfg["target_prompt"], "t")
        self.assertEqual(cfg["defense_type"], "jpeg_compression")

    def test_legacy_result_keys_present(self):
        out = normalize_results({"adversarial_final_tgt_cos": 0.7, "l_infinity": 0.03, "defense_type": "adaptive_reversal"})
        self.assertIn("adversarial_tgt_cos", out)
        self.assertIn("linf", out)
        self.assertIn("defense", out)


if __name__ == "__main__":
    unittest.main()
