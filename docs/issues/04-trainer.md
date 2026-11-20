---
id: 4
title: "[Infra] Training framework: Trainer, checkpoint/resume, train.py CLI"
owner: ege
labels: [area:infra, type:feature, priority:critical]
milestone: "M0 - Foundations"
week: W1
depends_on: []
---
## Goal
One training loop that every deep experiment uses, robust to Kaggle/Colab session limits.

## Context
Plan §3.2 (recipe), §4.3 (contract). `galaxy/utils/config.py` and `seed.py` are already implemented. Use them.
Develop against `tests/conftest.py` fixtures (random uint8 images) until #9 lands.

## Tasks
- [ ] `galaxy/training/optim.py`: `build_optimizer(model, cfg)` (AdamW, no weight decay on norm/bias params, optional separate head LR), `build_scheduler(opt, cfg, steps_per_epoch)` (linear warm-up → cosine, per step).
- [ ] `galaxy/training/trainer.py` `Trainer`:
  - [ ] Per batch: `GalaxyPreprocess` on GPU (`train=True` for train, `False` for eval) → model → loss. AMP (`torch.autocast` + `GradScaler`), optional grad clipping.
  - [ ] Validation every epoch: loss, accuracy, **macro-F1**. Early stopping on val macro-F1 (`patience`).
  - [ ] Checkpoints: `last.pt` every epoch (model, optimizer, scheduler, scaler, epoch, best metric, RNG states) and `best.pt`.
  - [ ] `--resume` continues exactly from `last.pt`.
  - [ ] `train_log.csv` per epoch (columns in plan §4.4) and a compact console log (tqdm optional).
  - [ ] `freeze_backbone: true` option (linear probe, needed by #21): only head params are trainable, and BN stays in eval mode.
  - [ ] Hooks for `losses.build_loss(cfg, class_counts)` and `samplers.build_sampler(cfg, labels)` (defaults: CE + label smoothing, shuffle). Eftelya implements the alternatives in #18.
- [ ] `scripts/train.py`: `--config`, `--seed`, `--set key=value ...`, `--resume`, `--no-test`. After training, load `best.pt` and call `evaluate()` (#5) on val and test.
- [ ] Log total train time, peak GPU memory and #params into the result JSON `extra`.
- [ ] `tests/test_trainer.py`: 2 epochs of a tiny model on synthetic data on CPU. Checks that the loss decreases, checkpoints exist, resume continues at the right epoch, and the same seed gives the same metrics.

## Deliverables
`trainer.py`, `optim.py`, `scripts/train.py`, tests, and a "How to train" section in the README.

## Acceptance criteria
- `python scripts/train.py --config configs/models/small_cnn.yaml --seed 0 --set train.max_epochs=1 data.fraction=0.1` runs end to end on Colab (after #9).
- If a run is killed and resumed, it reaches the same final epoch count, and its metrics are close to an uninterrupted run.
- CI is green.

## Blocks
#13, #15, #17, #18, #19, #20, #21
