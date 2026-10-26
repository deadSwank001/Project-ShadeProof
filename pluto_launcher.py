"""Pluto-IDE friendly launcher for ShadeProof defensive workflows."""

import argparse
import json

from shadeproof_v2.pluto import run_pluto_profile


def main():
    parser = argparse.ArgumentParser(description="Run ShadeProof with Pluto profile.")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--defense-strength", type=float)
    parser.add_argument("--epsilon", type=float)
    args = parser.parse_args()

    overrides = {}
    if args.iterations is not None:
        overrides["iterations"] = args.iterations
    if args.defense_strength is not None:
        overrides["defense_strength"] = args.defense_strength
    if args.epsilon is not None:
        overrides["epsilon"] = args.epsilon

    result = run_pluto_profile(overrides)
    printable = {k: v for k, v in result.items() if isinstance(v, (int, float, str, bool))}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
