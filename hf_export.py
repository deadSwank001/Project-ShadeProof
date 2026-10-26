"""Export Project-ShadeProof v2 model artifacts for Hugging Face."""

import argparse
import os

from configuration_shadeproof import ShadeProofConfig
from modeling_shadeproof import ShadeProofSanitizer


def build_model(defense_type: str, defense_strength: float, profile: str) -> ShadeProofSanitizer:
    config = ShadeProofConfig(
        defense_type=defense_type,
        defense_strength=defense_strength,
        runtime_profile=profile,
    )
    return ShadeProofSanitizer(config=config)


def write_readme(export_dir: str, config: ShadeProofConfig):
    path = os.path.join(export_dir, "README.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Project-ShadeProof v2\n\n")
        f.write("Defensive-first sanitizer model for poisoning reversal and detection.\n\n")
        f.write("## Config\n")
        f.write(f"- defense_type: `{config.defense_type}`\n")
        f.write(f"- defense_strength: `{config.defense_strength}`\n")
        f.write(f"- runtime_profile: `{config.runtime_profile}`\n")


def main():
    parser = argparse.ArgumentParser(description="Export ShadeProof v2 model for HF.")
    parser.add_argument("--export-dir", default="./hf_model")
    parser.add_argument("--defense-type", default="adaptive_reversal")
    parser.add_argument("--defense-strength", type=float, default=1.0)
    parser.add_argument("--runtime-profile", default="default")
    args = parser.parse_args()

    model = build_model(args.defense_type, args.defense_strength, args.runtime_profile)
    os.makedirs(args.export_dir, exist_ok=True)
    model.save_pretrained(args.export_dir)
    write_readme(args.export_dir, model.config)

    print(f"[+] Export complete: {args.export_dir}")


if __name__ == "__main__":
    main()
