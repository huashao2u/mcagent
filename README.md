# MCAgent

Minimal phase-one research prototype for action calibration with a tool-native action space:
`ANSWER`, `SEARCH`, `CALCULATE`, `CLARIFY`, `REFUSE`.

This implementation follows `mcagent_codex_spec.md` and focuses on a runnable closed loop:

- unified dataset loading
- prompt building and JSON decision parsing
- sandboxed tools
- rollout logging
- semantic/process tags
- oracle and local utility scoring
- pair construction for Step-DPO
- lightweight evaluation
- a smoke-testable DPO entrypoint

## Quick Start

Use the existing `salra` conda environment:

```bash
conda run -n salra python -m src.data.loaders --report
conda run -n salra python -m src.rollout.generate_rollouts --config configs/default.yaml --limit-per-dataset 3
conda run -n salra python -m src.pairs.build_pairs --config configs/default.yaml
conda run -n salra python -m src.training.run_dpo --config configs/default.yaml --smoke
conda run -n salra python -m src.eval.aggregate_results --config configs/default.yaml
```

## Notes

- `models/` is a symlink to local checkpoints under `/media/songyl/SALRA/models`.
- `competition_math` and `MintQA-Ti-v0.1` currently contain Git LFS pointer files instead of real parquet shards; the loaders detect and report this explicitly.
- The rollout engine supports both `heuristic` and `hf` backends. Phase-one smoke tests use `heuristic` by default so the full pipeline can run quickly.
