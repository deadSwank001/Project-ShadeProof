# Project-ShadeProof v2.0

Defensive-first rewrite of Project-ShadeProof focused on reversing poisoning-style perturbations and preserving backward compatibility with the prior rewrite.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22811993.svg)](https://doi.org/10.5281/zenodo.22811993)

## v2 Goals

- Reverse early poisoning-style effects through adaptive sanitization.
- Preserve existing API contracts (`backend.py`, GUI worker flow, model/config surfaces).
- Add Pluto-IDE compatible execution path.
- Keep attack simulation as evaluation tooling, not default deployment behavior.

## Repository Layout (v2)

```text
Project-ShadeProof/
├── backend.py                    # Stable public API + Qt worker wrapper
├── main.py                       # Existing PyQt6 GUI entrypoint (unchanged interface)
├── benchmark.py                  # Headless benchmark CLI
├── pluto_launcher.py             # Pluto-IDE friendly launcher
├── configuration_shadeproof.py   # Backward-compatible config surface
├── modeling_shadeproof.py        # Backward-compatible model output contract
├── hf_export.py                  # Model export utility
├── shadeproof_v2/
│   ├── metrics.py                # Metrics layer
│   ├── defenses.py               # Defense/sanitization layer
│   ├── features.py               # Surrogate feature backbones
│   ├── attacks.py                # Attack simulation layer
│   ├── orchestrator.py           # End-to-end orchestration
│   ├── compat.py                 # v1 compatibility adapters
│   └── pluto.py                  # Pluto runtime profile
└── tests/
    ├── test_compatibility.py
    └── test_defense_pipeline.py
```

## Backward Compatibility (v1 rewrite)

Guaranteed compatibility targets:

- `backend.py` exports remain stable (`AcademicMetrics`, `SanitizationDefensePipeline`, `SurrogateFeatureModel`, `SurrogateAttackEngine`, `ExperimentWorker`, `run_headless_benchmark`).
- Legacy config keys are accepted and mapped:
  - `eps -> epsilon`
  - `steps -> iterations`
  - `source -> source_prompt`
  - `target -> target_prompt`
  - `defense -> defense_type`
- Legacy result keys remain available:
  - `adversarial_tgt_cos`
  - `linf`
  - `defense`

## Pluto-IDE Usage

Run with Pluto-oriented defaults:

```bash
python pluto_launcher.py
```

Optional overrides:

```bash
python pluto_launcher.py --iterations 50 --defense-strength 1.5 --epsilon 0.0235
```

## GUI Usage

```bash
python main.py
```

## Headless Benchmark Usage

```bash
python benchmark.py --iterations 50 --defense adaptive_reversal
```

## Hugging Face Export

```bash
python hf_export.py --export-dir ./hf_model --defense-type adaptive_reversal
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Release Metadata

Zenodo archival metadata is configured in `.zenodo.json` and used for DOI-associated release metadata.

## License

Apache License 2.0 (see `LICENSE.txt`).
