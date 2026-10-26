"""Headless benchmark CLI for Project-ShadeProof v2."""

import argparse
import json

from backend import run_headless_benchmark


def main():
    parser = argparse.ArgumentParser(description="Run ShadeProof headless benchmark.")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--epsilon", type=float, default=8.0 / 255.0)
    parser.add_argument("--defense", default="adaptive_reversal")
    parser.add_argument("--source-prompt", default="a photograph of a dog")
    parser.add_argument("--target-prompt", default="a luxury leather handbag")
    parser.add_argument("--output-dir", default="./results")
    args = parser.parse_args()

    results = run_headless_benchmark(
        iterations=args.iterations,
        epsilon=args.epsilon,
        defense=args.defense,
        source_prompt=args.source_prompt,
        target_prompt=args.target_prompt,
        output_dir=args.output_dir,
    )

    printable = {k: v for k, v in results.items() if isinstance(v, (int, float, str, bool))}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
