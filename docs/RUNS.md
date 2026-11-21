# Run registry

Add a row **before** you start a run, and update `status` when it finishes. This prevents duplicate runs and spreads
the GPU quota across accounts (PROJECT_PLAN §7). Maintained by everyone, set up in #10.

| experiment | run_name | seeds | owner | platform / account | status | checkpoint location | commit |
|------------|----------|-------|-------|--------------------|--------|---------------------|--------|
| model_family | small_cnn (smoke test) | 0 | Berkay | Kaggle / berkay | planned | – | – |

Status values: `planned` · `running` · `done` · `failed (reason)` · `results merged (#PR)`.

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
| model | input | batch | time / epoch (T4, AMP) | peak memory | measured by |
|-------|-------|-------|------------------------|-------------|-------------|
| | | | | | |
