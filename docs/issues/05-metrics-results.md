---
id: 5
title: "[Eval] Metrics, prediction files & results schema (evaluate.py)"
owner: ege
labels: [area:evaluation, type:feature, priority:critical]
milestone: "M0 - Foundations"
week: W1
depends_on: []
---
## Goal
One evaluation code path and one result format for **all** models, including classical ones.

## Context
Plan §3.4, §4.4 (exact JSON schema and folder layout).

## Tasks
- [ ] `galaxy/evaluation/metrics.py`: `compute_metrics(y_true, y_pred_or_probs) -> dict`: accuracy, macro-F1, balanced accuracy, per-class recall/precision/F1, confusion matrix, n_samples. Works with 10 classes even if a class is missing from a subset.
- [ ] `galaxy/evaluation/results.py`:
  - [ ] `run_dir(experiment, run_name, seed) -> Path`
  - [ ] `save_eval(run_dir, eval_name, metrics, probs, y_true, orig_index, config, extra)` writes `eval_<name>.json` (schema in §4.4, including `git_commit` from `galaxy.utils.git_commit()` and `split_version`) and `preds_<name>.npz` (probs as float16)
  - [ ] `load_evals(glob="results/runs/**/eval_*.json") -> pandas.DataFrame` (one row per file, flattened metrics)
- [ ] `galaxy/evaluation/predict.py`: `predict(model, loader, preprocess, device, transform_fn=None) -> (probs, y_true, orig_index)`. `transform_fn` is a hook applied to raw uint8 batches **before** preprocessing. TTA (#17), rotation (#23) and corruptions (#22) plug in here.
- [ ] `scripts/evaluate.py`: `--run-dir <path> --eval test [--tta8] [--rotate 45] [--corruption blur --severity 2]`. Loads `best.pt` and the stored config, then writes `eval_*.json`. Implement the plain `test`/`val` path now; the others call into modules from #17/#22/#23 and can raise `NotImplementedError` until those land.
- [ ] `tests/test_metrics.py`: hand-computed examples, a missing-class case, and a schema round-trip (save → load → equal).

## Deliverables
`metrics.py`, `results.py`, `predict.py`, `scripts/evaluate.py`, tests, and `results/README.md` (update it if the schema changes).

## Acceptance criteria
- Metrics match `sklearn.metrics` on random inputs (test asserts this).
- A classical model (#14) can call `save_eval` with only numpy arrays, without torch.
- Re-evaluating an existing run produces an identical JSON, except for the timestamp.

## Blocks
#14, #16, #17, #22, #23, #25 and all training runs.
