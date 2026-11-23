# Run registry

Add a row **before** you start a run, and update `status` when it finishes. This prevents duplicate runs and spreads
the GPU quota across accounts (PROJECT_PLAN §7). Maintained by everyone, set up in #10.

| experiment | run_name | seeds | owner | platform / account | status | checkpoint location | commit |
|------------|----------|-------|-------|--------------------|--------|---------------------|--------|
| smoke | small_cnn (#6; 1 epoch, 10 % data) | 0 | Berkay | Kaggle / berkay | done | not kept (smoke test) | e004b4e |
| smoke | resnet50, convnext_tiny, vit_small (#12; 2 epochs, 100 % data) | 0 | Berkay | Kaggle / berkay | done | not kept (smoke test) | e004b4e |
| debug | small_cnn (runner dry run, #10, example row) | 0 | Ege | Kaggle / egeural | planned | Kaggle output of `kaggle-runner` v1 → `checkpoints/debug/small_cnn/seed0/` | – |

Status values: `planned` · `running` · `done` · `failed (reason)` · `results merged (#PR)`.

* **seeds:** `0,1,2` for a full batch. Use one row per batch, not one per seed.
* **checkpoint location:** where `best.pt` / `last.pt` can be found later. Kaggle: *notebook name + version* (the
  `checkpoints/` folder of its Output). Colab: `Drive/<DRIVE_DIR>/checkpoints/...` and whose Drive it is. Shared runs
  (#13, #19): the team checkpoint dataset.
* **commit:** the short hash the runner prints (`running commit ...`). It is also stored in every `eval_*.json`.

## Input pipeline throughput (#9, Kaggle Tesla T4, 4 CPUs, batch 64)
| setting | loader only | loader + GPU preprocess | preprocess only |
|---------|------------:|------------------------:|----------------:|
| rotflip → 224, memmap, 4 workers | 5154 | 4953 | 11393 img/s |
| rotflip → 224, in RAM, 2 workers | 4581 | 4557 | 11367 img/s |
| rotflip → 224, in RAM, 4 workers | 5496 | **5218** | 11339 img/s |
| none → 224, in RAM, 4 workers | 5569 | 5021 | 17659 img/s |
| rotflip → 128, in RAM, 4 workers | 5473 | 5218 | 13511 img/s |

About 5000 img/s ≫ the 1500 target and far above what any of our models trains at, so **the input pipeline is never the
bottleneck**. Defaults are fine: `num_workers: 4`; `in_memory: true` gains about 5 % on Kaggle.

## Smoke-test / cost measurements
| model | input | batch | time / epoch (T4, AMP) | peak memory | val acc / macro-F1 | measured by |
|-------|-------|-------|------------------------|-------------|--------------------|-------------|
| small_cnn | 128 | 64 | 11 s (on 10 % of the data) | 1.29 GB | 0.265 / 0.151 (1 epoch, 10 % data) | Berkay, #6 |
| resnet50 | 224 | 64 | 55 s | 3.31 GB | 0.851 / 0.833 (2 epochs) | Berkay, #12 |
| convnext_tiny | 224 | 64 | 75 s (63 s after the first epoch) | 4.58 GB | 0.844 / 0.817 (2 epochs) | Berkay, #12 |
| vit_small | 224 | 64 | 43 s | 2.62 GB | 0.839 / 0.816 (2 epochs) | Berkay, #12 |

Single Kaggle T4, `rotflip`, `data.in_memory=true`, commit e004b4e, seed 0. Time per epoch is the mean over the 2
epochs (full-data models: 192 steps/epoch); the first ConvNeXt epoch includes cuDNN autotuning. Peak memory is
`peak_mem_gb` of `train_summary.json` at batch 64. All three pretrained backbones are far above the 60 % val-accuracy
bar after 2 epochs. The largest batch size that fits a T4 was not measured: at batch 64 the peak is at most 4.6 GB of
the 15 GB, so 128 should fit for all three.
