# Run registry

Add a row **before** you start a run, and update `status` when it finishes. This prevents duplicate runs and spreads
the GPU quota across accounts (PROJECT_PLAN §7). Maintained by everyone, set up in #10.

| experiment | run_name | seeds | owner | platform / account | status | checkpoint location | commit |
|------------|----------|-------|-------|--------------------|--------|---------------------|--------|
| model_family | small_cnn (smoke test) | 0 | Berkay | Kaggle / berkay | planned | – | – |

Status values: `planned` · `running` · `done` · `failed (reason)` · `results merged (#PR)`.

## Smoke-test / cost measurements
| model | input | batch | time / epoch (T4, AMP) | peak memory | measured by |
|-------|-------|-------|------------------------|-------------|-------------|
| | | | | | |
