# Does Orientation Matter? Galaxy Morphology Classification on Galaxy10 DECaLS

**CS464 Machine Learning · Group 10** · Eren Kerimoğlu · Ege Ural · Yunus Dildar Çelik · Berkay Ataman · Eftelya Çoban

We classify 17,736 Galaxy10 DECaLS images (256×256, g/r/z) into 10 morphology classes. Our main question is how much
**building rotation invariance into a model** helps, compared with **changing the model family**. The invariance
options are rotation/flip augmentation, 8-view test-time augmentation and a rotation-equivariant CNN. The model families
are classical ML on hand-crafted features, CNNs and Vision Transformers. We also study transfer learning, data
efficiency, class imbalance, robustness to weaker-telescope conditions, and model errors.

| Document | What it contains |
|----------|------------------|
| [Proposal](docs/proposal/CS464_Galaxy_Proposal_Group10.pdf) | The original project proposal |
| [**Project plan**](docs/PROJECT_PLAN.md) | Requirements, fixed protocol, architecture & contracts, experiment specs, compute budget, timeline, risks |
| [**Work separation**](docs/WORK_SEPARATION.md) | Roles, issues per member, Gantt chart, dependency graph |
| [Issue texts](docs/issues/) | Source of every GitHub issue (tasks + acceptance criteria) |
| [Contributing](CONTRIBUTING.md) | Git workflow, code style, results workflow, rules |
| [Run registry](docs/RUNS.md) | Who is running what, where |

## Repository layout
```
galaxy/        importable package: data, features, models, training, evaluation, analysis, utils
configs/       YAML configs (base.yaml → models/*.yaml → experiments/*/*.yaml)
scripts/       CLI entry points: prepare_data, train, evaluate, tune, train_classical, aggregate_results, ...
notebooks/     EDA + Kaggle/Colab runner notebooks
tests/         pytest (synthetic data, runs in CI)
results/       committed metrics/predictions, generated tables and figures
report/        final report + slides
docs/          plan, work separation, issues, run registry
tools/github/  script that creates the GitHub labels, milestones and issues
```

## Setup
```bash
git clone https://github.com/ErenKerimoglu/CS464_Project.git
cd CS464_Project
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # light: enough for tests on CPU
pip install -e ".[all,dev]"        # full: deep + classical + tuning + escnn (GPU machine / Kaggle / Colab)
pytest && ruff check .
```

For the equivariant models, `escnn==1.0.11` is pinned. On Windows, install Microsoft C++ Build Tools if the
`lie-learn` transitive dependency needs to compile (CPU test coverage is otherwise available through
`pytest.importorskip("escnn")`). Kaggle and Colab Linux environments are the intended GPU runtime.

## How to train
```bash
# 1-epoch smoke test on 10 % of the training data (always do this before a 3-seed batch)
python scripts/train.py --config configs/models/small_cnn.yaml --seed 0 --set train.max_epochs=1 data.fraction=0.1

# a real run; any config value can be overridden with --set key=value
python scripts/train.py --config configs/models/resnet50.yaml --seed 0 --set train.lr=3e-4

# the session died (Kaggle 12 h limit)? Run the same command with --resume: it continues exactly from last.pt
python scripts/train.py --config configs/models/resnet50.yaml --seed 0 --set train.lr=3e-4 --resume
```
* Each run is identified by `experiment` / `run_name` from the config plus `--seed`. Checkpoints go to
  `checkpoints/<experiment>/<run_name>/seed<k>/{last,best}.pt` (gitignored; set `GALAXY_CKPT_DIR` to put them on
  Drive or `/kaggle/working`). Results go to `results/runs/<experiment>/<run_name>/seed<k>/`: `train_log.csv`,
  `train_summary.json`, and `eval_val.json` / `eval_test.json` with `preds_*.npz` (schema: [results/README.md](results/README.md)).
* Recipe (PROJECT_PLAN §3.2): AdamW (no weight decay on norm/bias), linear warm-up → per-step cosine, AMP on GPU,
  gradient clipping, early stopping on **val** macro-F1, `best.pt` = best val macro-F1. The test set is evaluated
  once, after training, with `best.pt` (`--no-test` skips it, e.g. while tuning).
* `model.freeze_backbone: true` trains only the classifier head (linear probe); `loss.name` / `sampler.name` switch
  the loss and sampler (E4). `train.head_lr` gives the head its own learning rate.
* A run refuses to start if its `last.pt` already exists. Pass `--resume`, or delete the checkpoint folder.
* Re-evaluate a finished run (also TTA / rotation / corruption variants as they land):
  `python scripts/evaluate.py --run-dir results/runs/<experiment>/<run_name>/seed0 --eval test`.

## Reproducing results (filled in as the issues land)
```bash
python scripts/prepare_data.py                                          # #2, #3: download, dedup, split, cache
python scripts/extract_features.py --features hog color morph            # #7, #11
python scripts/train_classical.py --model xgboost --tune --n-trials 50   # #14
python scripts/train.py --config configs/models/resnet50.yaml --seed 0   # #4
python scripts/evaluate.py --run-dir results/runs/<exp>/<run>/seed0 --eval test --tta8   # #5, #17
python scripts/aggregate_results.py                                     # #16 → results/tables/
```

## Status
Progress is tracked with GitHub issues and milestones (M0 Foundations → M4 Report). See
[WORK_SEPARATION.md](docs/WORK_SEPARATION.md).
