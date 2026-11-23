# TA feedback on the proposal (23 Nov 2026) and how we follow it

The proposal was graded **5/5**. This file checks each point of the feedback against our plan
([PROJECT_PLAN.md](PROJECT_PLAN.md)). It records what we changed and where we deliberately keep our own approach,
so we can explain it in the progress report.

## Point-by-point

| # | TA feedback | Our plan before the feedback | Decision |
|---|-------------|------------------------------|----------|
| 1 | Use **macro-F1 to select the best model**, because the classes are imbalanced. Report accuracy and per-class recall as extra scores. | Already the protocol (§3.2–3.4). Val macro-F1 drives early stopping, best-checkpoint selection and the Optuna objective (deep #15 and classical #14). We report accuracy, macro-F1, balanced accuracy, per-class recall/precision and confusion matrices (#5). | **No change**, already aligned. |
| 2 | The workload is large. **Focus on the orientation experiment** as the main one, plus **one or two others** (e.g. transfer learning, class imbalance). The rest can be optional. | 6 experiments (E1–E6) + SDSS stretch, about 75 full runs. | **Adopted.** Core: E1 (orientation, incl. escnn and the classical orientation sweep) with the family comparison it needs. Additional: **E2 transfer learning** and **E4 class imbalance**. Optional, after a progress presentation: E3 data efficiency, E5 robustness, the Grad-CAM/t-SNE part of E6, SDSS. About 48 full runs now. See plan §6 *Scope tiers*. |
| 3 | Fine-tuning several pretrained networks with **three seeds** is a large workload. | 3 seeds for every deep run. | **Partly kept (our choice).** Core and additional runs keep 3 seeds; optional experiments drop to 1 seed. Reason: measured cost is low (ResNet-50 55 s/epoch on a T4, about 30 min per 30-epoch run, `docs/RUNS.md`), and the differences we expect between augmentation policies are a few points. Without mean ± std over seeds (plus McNemar / paired bootstrap, §3.4), we could not say whether they are real. |
| 4 | **Do not postpone experiments** to the last weeks; training takes time. | Core experiments planned for 7–20 Dec. | **Adopted.** Timeline pulled forward (plan §8): progress-report runs this week, core experiments by 13 Dec, additional by 27 Dec. |
| 5 | Use **validation data for hyperparameter tuning**. Compare the final models on the **same test set**. This will be checked in the final report. | Already the protocol (NFR-2, NFR-3). | **No change**, already enforced. Details below. |

## How we guarantee point 5 (for the report)
* **One frozen split**: `data/splits/split_v1.csv` (stratified 70/15/15, seed 464), committed to git, never
  regenerated. `prepare_data.py` refuses to overwrite it and checks that a rebuild reproduces it exactly. Every
  result file stores `split_version: v1` and the git commit.
* **Same 2,643 test galaxies for every model**, deep and classical alike. Both use the same `compute_metrics`
  (#5), and every config is stored inside its result JSON.
* **Validation only** for Optuna, early stopping and choosing the checkpoint. The test set is read once, after
  training ends, by `evaluate()`. Classical models also tune on val (`train_classical.py --tune`), and their PCA and
  scalers are fitted on train only.
* **No leakage through duplicates**: deduplication by sky position happens *before* the split (#3).

## Choices worth mentioning in the progress report
* **Deduplication finding:** all 59 coordinate duplicates are pixel-identical images with *two different labels*.
  We drop both copies (118 images), so our numbers are on 17,618 galaxies and not directly comparable to papers that
  use all 17,736.
* **Rotation-safe geometry:** rotate the full 256 px image, then center-crop 180 px (< 256/√2), then resize. Every
  angle is free of padding artifacts, so the rotation sweep measures orientation and not border effects.
* **Per-band normalization** from our training set (g, r, z), also for the ImageNet-pretrained models, instead of
  ImageNet RGB statistics.
* **GPU-side augmentation**, so data loading is never the bottleneck (about 5,000 img/s on a Kaggle T4).
* **Status of the equivariant CNN:** the model is implemented and tested, but `escnn` does not install on Kaggle's
  current Python 3.13. Runs go to Colab or an older Kaggle image (#10).
