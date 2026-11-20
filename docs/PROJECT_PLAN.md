# Project Plan: Does Orientation Matter?
### Galaxy Morphology Classification on Galaxy10 DECaLS (CS464, Group 10)

This document turns the [proposal](proposal/CS464_Galaxy_Proposal_Group10.pdf) into requirements, fixed design
decisions, and an implementation plan. Who does what, and when, is in [WORK_SEPARATION.md](WORK_SEPARATION.md).
Each work item is a GitHub issue; the issue texts are also kept in [`docs/issues/`](issues/).

> **Rule of thumb:** if this document and an issue disagree, raise it in the issue thread and fix the document.
> Everyone works against the **contracts** in §4. Do not change one silently, because other people's code depends on it.

---

## 0. Contents
1. [Research questions & hypotheses](#1-research-questions--hypotheses)
2. [Requirements](#2-requirements)
3. [Fixed design decisions (the protocol)](#3-fixed-design-decisions-the-protocol)
4. [Software architecture & contracts](#4-software-architecture--contracts)
5. [Implementation plan: topic → aspect → implementation](#5-implementation-plan-topic--aspect--implementation)
6. [Experiment specifications & run matrix](#6-experiment-specifications--run-matrix)
7. [Compute budget](#7-compute-budget)
8. [Timeline & milestones](#8-timeline--milestones)
9. [Risks & mitigations](#9-risks--mitigations)
10. [Definition of done (project level)](#10-definition-of-done-project-level)

---

## 1. Research questions & hypotheses

| ID | Question | Hypothesis we test |
|----|----------|--------------------|
| **RQ1 (main)** | How much does building in rotation invariance improve accuracy, macro-F1 and robustness, compared with changing the model family? | Rotation/flip augmentation and 8-view TTA give a consistent gain. An equivariant CNN matches augmented CNNs with fewer parameters and stays flat under test-time rotation. The gain from invariance is comparable to the gap between CNN families. |
| RQ2 | Does ImageNet pretraining help on astronomical images? | Full fine-tuning > linear probe > from scratch, and the gap is larger for ViT than for CNNs. |
| RQ3 | How data-efficient is each approach? | Equivariant and pretrained models degrade least at 10–25 % of the data. |
| RQ4 | How do rare and visually similar classes affect each approach? | Imbalance methods raise recall on *Cigar-shaped Smooth* (334 images) and *Disturbed*. Barred vs. Unbarred Spiral confusion stays the main error source. |
| RQ5 | How robust are the models to weaker-telescope conditions? | Blur and downsampling hurt ViT and HOG-based models most. Morphology statistics are the most stable features. |
| RQ6 | What do models look at? | Grad-CAM mass mostly lies on the target galaxy. Errors correlate with attention on background stars or companions. |

**Headline deliverable:** one figure of **test accuracy vs. rotation angle** for all model families (classical,
SmallCNN, ResNet-50 ± augmentation ± TTA, ConvNeXt-T, ViT-S/16, equivariant CNN), and one **main results table**
(accuracy, macro-F1, mean ± std over 3 seeds).

---

## 2. Requirements

### 2.1 Functional requirements

| ID | Requirement | Source (proposal) | Issue(s) |
|----|-------------|-------------------|----------|
| FR-D1 | Download Galaxy10 DECaLS (17,736 images, 256×256×3, g/r/z) automatically and verify its integrity. | Data | #2 |
| FR-D2 | Remove duplicate galaxies by sky coordinates (RA/Dec). | Preprocessing | #3 |
| FR-D3 | One fixed stratified **70/15/15** train/val/test split, committed to git and shared by everyone. | Preprocessing | #3 |
| FR-D4 | Per-band standardization (train statistics), center crop, resize to 224×224 for pretrained networks. | Preprocessing | #9 |
| FR-F1 | HOG features and color histograms. | Preprocessing | #7 |
| FR-F2 | Morphology statistics (concentration, asymmetry, Gini, M20, and others) with **statmorph**. | Preprocessing | #11 |
| FR-F3 | PCA reduction of the hand-crafted features. | Preprocessing | #14 |
| FR-M1 | Logistic Regression, SVM (RBF), XGBoost on hand-crafted features. | Model selection | #14 |
| FR-M2 | Small CNN trained from scratch. | Model selection | #6 |
| FR-M3 | Fine-tuned ImageNet-pretrained ResNet-50, ConvNeXt-Tiny, ViT-S/16. | Model selection | #12 |
| FR-M4 | Rotation-equivariant CNN with **escnn**. | Exp. 1 | #8, #13 |
| FR-T1 | Hyperparameter tuning with **Optuna**, on the validation set only. | Experiments | #14, #15 |
| FR-E1 | Orientation: no aug vs. rotation/flip aug vs. 8-view TTA vs. equivariant CNN, plus accuracy vs. test rotation angle. | Exp. 1 | #17, #23, #24 |
| FR-E2 | Transfer learning: scratch vs. linear probe vs. full fine-tuning. | Exp. 2 | #21 |
| FR-E3 | Data efficiency: learning curves at 10/25/50/100 % of training data. | Exp. 3 | #20, #24 |
| FR-E4 | Class imbalance: weighted CE, focal loss, oversampling, compared by macro-F1. | Exp. 4 | #18 |
| FR-E5 | Robustness: accuracy drop under noise, blur, downsampling. | Exp. 5 | #22 |
| FR-E6 | Error analysis: confusion matrices, Grad-CAM, t-SNE. | Exp. 6 | #25 |
| FR-R1 | Report accuracy, macro-F1, per-class recall. Deep models: 3 seeds, mean ± std. | Other | #5, #16 |
| FR-R2 | *(Stretch)* Evaluate the best models on the low-resolution SDSS Galaxy10. | Other | #26 |
| FR-R3 | Final report and presentation. | Course | #27 |

### 2.2 Non-functional requirements

| ID | Requirement | How we meet it |
|----|-------------|----------------|
| NFR-1 Reproducibility | Every number in the report can be regenerated from a config + seed + git commit. | YAML configs, `seed_everything`, git hash stored in every result file (§4.4). |
| NFR-2 No leakage | The test set is never used for tuning, early stopping or model choice. | Early stopping and Optuna use **val** only. Test evaluation is a separate step. Dedup happens **before** splitting. |
| NFR-3 Fair comparison | All models see the same split, crop, resolution policy and evaluation code. | Shared `GalaxyDataset`, `GalaxyPreprocess`, `evaluate()` (§4). |
| NFR-4 Runs on free GPUs | Training fits on a Kaggle T4/P100 (16 GB) or Colab T4, and survives session limits. | AMP, batch 64, checkpoint + resume every epoch, GPU-side augmentation. |
| NFR-5 Parallel teamwork | Five people work at once without blocking each other. | Contracts frozen in week 1, synthetic-data unit tests, one owner per module. |
| NFR-6 Code quality | Readable, tested, consistent. | `ruff` + `pytest` in CI, PR review required, type hints on public functions. |
| NFR-7 Data hygiene | No datasets or checkpoints in git. | `.gitignore`. Data lives on Kaggle Datasets / Google Drive. Only metrics JSON and small prediction files are committed. |

---

## 3. Fixed design decisions (the protocol)

These are decided now so results are comparable. Changing one needs agreement from the team in an issue.

### 3.1 Data
| Topic | Decision |
|-------|----------|
| Source | `Galaxy10_DECals.h5` from astroNN (Zenodo record 10845026, about 2.5 GB). Keys: `images` (N,256,256,3 uint8), `ans`, `ra`, `dec`, `redshift`, `pxscale`. |
| Classes | 0 Disturbed (1081) · 1 Merging (1853) · 2 Round Smooth (2645) · 3 In-between Round Smooth (2027) · 4 Cigar Shaped Smooth (334) · 5 Barred Spiral (2043) · 6 Unbarred Tight Spiral (1829) · 7 Unbarred Loose Spiral (2628) · 8 Edge-on without Bulge (1423) · 9 Edge-on with Bulge (1873) |
| Deduplication | `astropy` sky matching with a **1″** radius (≈4 px at 0.262″/px). Report counts at 0.5/1/2/5″ and visually inspect a sample of pairs. Same label: keep the lowest index. **Conflicting labels: drop all copies.** Store the removed indices and the reason. |
| Split | `sklearn` stratified 70/15/15, `random_state=464`, **after** dedup. Saved as `data/splits/split_v1.csv` (columns: `orig_index, label, ra, dec, split`). **Committed to git. Never regenerated.** |
| Normalization | Per-band mean/std computed on **train** pixels of the center crop, in [0,1] scale. Saved to `data/splits/band_stats_v1.json` (committed). |
| Geometry | Rotate the **full 256×256** image first, then **center-crop 180×180**, then resize to the model input (224 for pretrained models). 180 < 256/√2 ≈ 181, so a rotation by any angle leaves no empty corners in the crop. We use the same crop for every model and policy, so augmentation is the only thing that changes. #2 checks that galaxies fit inside the 180 px crop. |
| Storage | `scripts/prepare_data.py` writes `data/processed/images_256.npy` (uint8, memory-mappable) and `data/processed/meta.csv`. The cache is uploaded once as a private Kaggle Dataset (and to Drive for Colab). |

### 3.2 Training (deep models), default recipe
| Item | Default (tuned per model in #15) |
|------|-----------------------------------|
| Optimizer | AdamW, weight decay 0.05 |
| LR | 1e-4 pretrained fine-tuning · 1e-3 scratch / linear probe · cosine schedule, 1 warm-up epoch |
| Epochs | 30 pretrained · 60 scratch (SmallCNN, escnn, scratch ResNet/ViT) · early stopping, patience 8 on **val macro-F1** |
| Batch / precision | 64, AMP (fp16) |
| Loss | Cross-entropy, label smoothing 0.1 (E4 swaps the loss) |
| Augmentation policies | `none` (crop+resize only) · **`rotflip` (default)**: uniform random angle in [0°, 360°) + random h/v flips · `d4` *(optional)*: random element of the dihedral group (exact 90° rotations and flips) |
| Augmentation implementation | **On GPU**, batched (`affine_grid` + `grid_sample`), inside `GalaxyPreprocess`. The DataLoader only yields uint8 tensors, so CPU workers are never the bottleneck on Kaggle. |
| Seeds | `{0, 1, 2}` for all reported deep runs |
| Model selection | Checkpoint with the best val macro-F1 |

**Reference backbone.** Experiments E1a, E3, E4 and E5 train **ResNet-50 (pretrained, `rotflip`)** as the reference
model, so each experiment changes only one thing. The best overall model is also included in the evaluation-only parts
(TTA, rotation sweep, robustness, Grad-CAM).

**timm model names.** `resnet50.tv_in1k` · `convnext_tiny.fb_in1k` · `vit_small_patch16_224.augreg_in21k_ft_in1k`.

### 3.3 Classical models
| Item | Decision |
|------|----------|
| Input | The same 180×180 center crop. Raw [0,1] pixels, no standardization. |
| HOG | Grayscale (mean of bands) resized to 128×128. 9 orientations, 16×16 px cells, 2×2 blocks. |
| Color | Per-band 32-bin histograms + mean/std/percentiles of the g−r and r−z "color" images inside the galaxy mask. |
| Morphology | statmorph on the r-band with a photutils segmentation of the **central** source. Features: C, A, S, Gini, M20, Gini-M20 bulge/merger statistics, ellipticity, elongation, r20/r80, plus statmorph's `flag`. Failures are imputed with the train median, and a `morph_failed` indicator is added. |
| Reduction | StandardScaler → PCA keeping 95 % of variance, fitted on **HOG only**, then concatenated with the scaled color and morphology features. Fitted on train only (sklearn `Pipeline`). |
| Tuning | Optuna, 50 trials (TPE), objective = val macro-F1. |
| Seeds | XGBoost: 3 seeds. LR/SVM are deterministic: one run, report std = 0 and say so. |

### 3.4 Evaluation
| Item | Decision |
|------|----------|
| Metrics | accuracy, macro-F1, per-class recall, per-class precision, confusion matrix (raw + row-normalized) |
| Aggregation | mean ± std over seeds, as a markdown/CSV table generated by `scripts/aggregate_results.py` |
| Significance | For headline comparisons: McNemar's test on the pooled test predictions and paired bootstrap 95 % CI of the macro-F1 difference (#16) |
| TTA-8 | Average softmax over the 8 dihedral views (4 rotations × optional horizontal flip) of the **crop** |
| Rotation sweep | Test angles 0°, 15°, …, 345° (24 angles), applied to the 256 image before cropping (same as training geometry) |

---

## 4. Software architecture & contracts

```
CS464_Project/
├── galaxy/                      # importable package  (pip install -e .)
│   ├── constants.py             # class names, paths, split version  [done]
│   ├── utils/                   # seed, config, git hash               [done]
│   ├── data/                    # download, dedup, split, dataset, GPU preprocessing      → Eren
│   ├── features/                # HOG/color, segmentation, statmorph, feature cache       → Yunus
│   ├── models/                  # registry, SmallCNN, timm wrappers, escnn model         → Berkay, Eftelya
│   ├── training/                # Trainer, losses, samplers, optim/scheduler             → Ege (+ Eftelya: losses/samplers)
│   ├── evaluation/              # metrics, result I/O, TTA, rotation, corruptions        → Ege, Eren, Yunus
│   └── analysis/                # aggregation, stats tests, plots, Grad-CAM, t-SNE       → Ege, Eftelya
├── configs/                     # YAML: base.yaml + models/*.yaml + experiments/*.yaml
├── scripts/                     # CLI entry points (prepare_data, train, evaluate, tune, ...)
├── notebooks/                   # EDA + Kaggle/Colab runner notebooks (thin wrappers around scripts/)
├── tests/                       # pytest, synthetic data only; runs in CI on CPU
├── results/runs/                # metrics JSON + small prediction files (committed)
├── results/tables, figures/     # generated by scripts/aggregate_results.py and analysis code
├── report/                      # final report sources + presentation
└── docs/                        # this plan, work separation, issue texts, proposal
```

### 4.1 Data contract (`galaxy.data`, owner: Eren)
```python
GalaxyDataset(split: Literal["train","val","test"], fraction: float = 1.0, fraction_seed: int = 0)
    __getitem__(i) -> (image_uint8: Tensor[3,256,256], label: int, orig_index: int)
build_dataloaders(cfg) -> dict[str, DataLoader]

GalaxyPreprocess(nn.Module)            # runs on GPU, batched
    __init__(policy: str, out_size: int, crop: int = 180, band_stats: dict)
    forward(x_uint8: Tensor[B,3,256,256], train: bool) -> Tensor[B,3,out,out]  # float, standardized
rotate_batch(x: Tensor[B,C,H,W], angles_deg: Tensor[B]) -> Tensor          # used by aug, sweep, TTA
load_split() -> pandas.DataFrame          # reads data/splits/split_v1.csv
load_raw_images(indices) -> np.ndarray    # uint8 (N,256,256,3) from the processed cache
```
`fraction < 1` returns a **nested** stratified subset (10 % ⊂ 25 % ⊂ 50 %). The subset is fixed by `fraction_seed`;
the training seed varies independently.

### 4.2 Model contract (`galaxy.models`, owner: Berkay; escnn: Eftelya)
```python
build_model(name: str, num_classes: int = 10, pretrained: bool = True, **kwargs) -> nn.Module
#   names: "small_cnn", "resnet50", "convnext_tiny", "vit_small", "escnn_d4" (or "escnn_c8")
#   every model exposes:
model.forward(x) -> logits [B,10]
model.forward_features(x) -> embedding [B,D]   # for t-SNE / linear probe
model.gradcam_target_layer -> nn.Module          # for Grad-CAM (CNNs); ViT uses reshape_transform
model.input_size -> int                           # 224 for timm models, configurable for small_cnn / escnn
```

### 4.3 Training contract (`galaxy.training`, owner: Ege)
```bash
python scripts/train.py --config configs/experiments/e1_orientation/resnet50_none.yaml --seed 0 \
       [--set train.lr=3e-4 data.fraction=0.25]  [--resume]
```
* Writes `checkpoints/<experiment>/<run_name>/seed<k>/{last,best}.pt` (gitignored; copy to Drive/Kaggle output).
* After training, runs `evaluate()` on **val and test** with the best checkpoint and writes result files (§4.4).
* Plug-in points: `losses.build_loss(cfg, class_counts)`, `samplers.build_sampler(cfg, labels)`. Owned by Eftelya (#18), default = CE and shuffle.

### 4.4 Results contract (`galaxy.evaluation.results`, owner: Ege)
```
results/runs/<experiment>/<run_name>/seed<k>/
    eval_<eval_name>.json     # eval_name: "test", "val", "test_tta8", "test_rot045", "test_blur_s2", ...
    preds_<eval_name>.npz     # probs [N,10] float16, y_true [N], orig_index [N]
    train_log.csv             # epoch, train_loss, val_loss, val_acc, val_macro_f1, lr, time
```
`eval_*.json` schema:
```json
{
  "experiment": "e1_orientation", "run_name": "resnet50_rotflip", "seed": 0,
  "model": "resnet50", "eval_name": "test_tta8", "split": "test", "split_version": "v1",
  "git_commit": "abc1234", "timestamp": "2026-12-01T12:00:00",
  "config": { "...": "full resolved config" },
  "metrics": {
    "accuracy": 0.9, "macro_f1": 0.88, "balanced_accuracy": 0.88,
    "per_class_recall": [10 floats], "per_class_precision": [10 floats], "per_class_f1": [10 floats],
    "confusion_matrix": [[10x10 ints]], "n_samples": 2650
  },
  "extra": { "rotation_deg": 45, "corruption": "blur", "severity": 2 }
}
```
**Classical models write the same schema**, so aggregation, plots and stats tests work for every model family.

### 4.5 Experiment naming
| `experiment` | `run_name` examples |
|--------------|---------------------|
| `baselines_classical` | `logreg`, `svm_rbf`, `xgboost` |
| `model_family` | `small_cnn`, `resnet50`, `convnext_tiny`, `vit_small`, `escnn_d4` |
| `e1_orientation` | `resnet50_none`, `convnext_tiny_none`, `resnet50_d4` (rotflip runs are reused from `model_family`) |
| `e2_transfer` | `resnet50_scratch`, `resnet50_linear`, `vit_small_scratch`, `vit_small_linear` |
| `e3_data_eff` | `resnet50_f010`, `small_cnn_f025`, `escnn_d4_f050`, `xgboost_f010`, ... |
| `e4_imbalance` | `resnet50_wce`, `resnet50_focal`, `resnet50_oversample` |
| `e5_robustness` | evaluation-only: `eval_name = test_<corruption>_s<severity>` on existing runs |
| `sdss` | evaluation-only on the SDSS Galaxy10 |

---

## 5. Implementation plan: topic → aspect → implementation

Each row is one aspect. The issue number is where the work is specified in detail (tasks + acceptance criteria).

### T1 · Data foundation
| Aspect | Implementation | Issue | Owner |
|--------|----------------|-------|-------|
| Acquisition | `galaxy/data/download.py`: resumable download, MD5/size check, `h5py` reader | #2 | Eren |
| Exploration | `notebooks/00_eda.ipynb`: class counts, sample grid per class, pixel stats per band, galaxy-extent check for the 180 px crop, redshift distribution | #2 | Eren |
| Deduplication | `galaxy/data/dedup.py`: astropy `search_around_sky`, conflict handling, report | #3 | Eren |
| Split + stats | `galaxy/data/split.py`: stratified 70/15/15 → `split_v1.csv`, `band_stats_v1.json` (both committed) | #3 | Eren |
| Dataset + GPU preprocessing | `galaxy/data/dataset.py`, `galaxy/data/transforms.py`: memmap dataset, nested fractions, `GalaxyPreprocess`, `rotate_batch` | #9 | Eren |

### T2 · Infrastructure
| Aspect | Implementation | Issue | Owner |
|--------|----------------|-------|-------|
| Config + CLI | `galaxy/utils/config.py` (done: YAML merge + `--set` overrides), `scripts/train.py` | #4 | Ege |
| Trainer | `galaxy/training/trainer.py`: AMP, AdamW + cosine/warm-up, early stopping, best/last checkpoints, **resume**, CSV log, freeze-backbone option | #4 | Ege |
| Metrics + results I/O | `galaxy/evaluation/metrics.py`, `results.py`, `scripts/evaluate.py` | #5 | Ege |
| Cloud runners | `notebooks/kaggle_runner.ipynb`, `notebooks/colab_runner.ipynb`, private Kaggle Dataset with the processed cache, `docs/RUNS.md` (run registry: who runs what, where) | #10 | Ege |
| Aggregation + stats | `scripts/aggregate_results.py`, `galaxy/analysis/stats_tests.py`, `galaxy/analysis/plotting.py` (shared style, class-color map) | #16 | Ege |

### T3 · Classical ML
| Aspect | Implementation | Issue | Owner |
|--------|----------------|-------|-------|
| HOG + color | `galaxy/features/hog_color.py`, cached to `data/features/*.parquet` | #7 | Yunus |
| Segmentation + statmorph | `galaxy/features/segmentation.py` (also used by Grad-CAM analysis), `galaxy/features/morphology.py`, joblib-parallel, failure handling | #11 | Yunus |
| Models + tuning | `scripts/train_classical.py`: PCA pipeline, LR/SVM/XGB, Optuna, results in the §4.4 schema | #14 | Yunus |
| Orientation + data efficiency (classical) | Rotation sweep (features re-extracted on rotated test images) and learning curves | #24 | Yunus |

### T4 · Deep model families
| Aspect | Implementation | Issue | Owner |
|--------|----------------|-------|-------|
| Registry + SmallCNN | `galaxy/models/registry.py`, `small_cnn.py` (4 conv blocks, BN, GAP, about 1–2 M params) | #6 | Berkay |
| Pretrained backbones | `galaxy/models/timm_models.py`: ResNet-50, ConvNeXt-T, ViT-S/16, `forward_features`, Grad-CAM layers, linear-probe freezing | #12 | Berkay |
| Tuning | `scripts/tune.py`: generic Optuna harness (lr, wd, label smoothing, drop-path) with a MedianPruner | #15 | Berkay |
| Family comparison | 4 models × 3 seeds with tuned hyperparameters → main table | #19 | Berkay |

### T5 · Rotation equivariance
| Aspect | Implementation | Issue | Owner |
|--------|----------------|-------|-------|
| Architecture | `galaxy/models/equivariant.py`: escnn D4 (and C8) steerable CNN, invariant pooling head, equivariance unit tests | #8 | Eftelya |
| Training | Config, tuning, 3 seeds, parameter-count comparison | #13 | Eftelya |

### T6 · Experiments
| Exp. | Aspect | Implementation | Issue | Owner |
|------|--------|----------------|-------|-------|
| E1a | Augmentation + TTA | Train `none` vs `rotflip` (ResNet-50, ConvNeXt-T), `galaxy/evaluation/tta.py` (8-view), evaluate all deep models with TTA | #17 | Eren |
| E1b | Rotation sweep + synthesis | `galaxy/evaluation/rotation.py`, sweep for all deep models, combine with classical curves → headline figure + RQ1 answer | #23 | Eren |
| E2 | Transfer learning | Scratch vs. linear probe vs. full FT (ResNet-50, ViT-S/16) | #21 | Berkay |
| E3 | Data efficiency | Nested 10/25/50/100 % runs for ResNet-50, SmallCNN, escnn | #20 | Ege |
| E4 | Class imbalance | `galaxy/training/losses.py` (weighted CE, focal), `samplers.py` (oversampling) + runs | #18 | Eftelya |
| E5 | Robustness | `galaxy/evaluation/corruptions.py` (noise, blur, downsample × severities) + evaluation of all models | #22 | Yunus |
| E6 | Error analysis | Confusion matrices, Grad-CAM + "attention on target galaxy" score, t-SNE | #25 | Eftelya |
| Stretch | SDSS generalization | Evaluate the best models on Galaxy10 SDSS (69×69, upsampled) | #26 | Eftelya |

### T7 · Reporting
| Aspect | Implementation | Issue | Owner |
|--------|----------------|-------|-------|
| Final report + slides | `report/`: each owner writes the method + results section for their issues. Eren compiles and edits. | #27 | All |

---

## 6. Experiment specifications & run matrix

All deep runs: 3 seeds. "Reused" means the run already exists from another experiment and is **not** retrained.

| Exp. | Model(s) | Variants | New training runs | Evaluation-only work |
|------|----------|----------|-------------------|----------------------|
| Classical (#14) | LR, SVM-RBF, XGB | tuned | 1 + 1 + 3 | – |
| Family (#19) | SmallCNN, ResNet-50, ConvNeXt-T, ViT-S | `rotflip`, tuned | 4 × 3 = **12** | – |
| Escnn (#13) | escnn D4 | `rotflip` (and `none`) | 3 (+3) | – |
| E1a (#17) | ResNet-50, ConvNeXt-T | `none` (`rotflip` reused) | 2 × 3 = **6** (+3 `d4` optional) | TTA-8 on every deep run |
| E1b (#23, #24) | all | – | 0 | 24 angles × every model |
| E2 (#21) | ResNet-50, ViT-S | scratch, linear (full FT reused) | 2 × 2 × 3 = **12** | – |
| E3 (#20) | ResNet-50, SmallCNN, escnn | 10/25/50 % (100 % reused) | 3 × 3 × 3 = **27** (small) | – |
| E4 (#18) | ResNet-50 | wCE, focal, oversample (CE reused) | 3 × 3 = **9** | – |
| E5 (#22) | all | 3 corruptions × 3–4 severities | 0 | about 11 evaluations × every model |
| E6 (#25) | best model, SmallCNN, escnn | – | 0 | Grad-CAM, t-SNE |
| Tuning (#15) | 4 models (+escnn) | about 15 trials each, short schedule | about 60 short runs | – |

**Total:** about 75 full training runs + about 60 short tuning runs. See §7.

### Per-experiment outputs (what goes into the report)
| Exp. | Table | Figure |
|------|-------|--------|
| Family | Main table: model × {acc, macro-F1, #params, train time} | – |
| E1 | aug × TTA × model grid | **Accuracy vs. rotation angle** (all families); bar chart of the gain from each invariance mechanism |
| E2 | mode × model | Grouped bars |
| E3 | – | Learning curves (macro-F1 vs. % data, log x-axis, ±std band) |
| E4 | method × {acc, macro-F1, recall on the 3 rarest classes} | Per-class recall heatmap |
| E5 | mean corruption accuracy per model | Accuracy vs. severity, one panel per corruption |
| E6 | Grad-CAM on-galaxy score per model | Confusion matrices, Grad-CAM grid (correct vs. wrong), t-SNE |

---

## 7. Compute budget
* Rough cost: ResNet-50 at 224 px, about 12.4 k training images, T4 + AMP ≈ 1.5–2 min/epoch, so about 45 min for a 25-epoch run.
  ViT-S and ConvNeXt-T cost about the same. SmallCNN and escnn at 128 px are cheaper.
* About 75 full runs × about 40 min + tuning ≈ **60–70 GPU-hours**. Kaggle gives 30 GPU-h/week per account, so 5 accounts give
  about 150 h/week. **The budget fits, but only if runs are spread across all five accounts.** Log every run in `docs/RUNS.md` (#10).
* Kaggle sessions stop after 12 h. The Trainer checkpoints every epoch and resumes with `--resume`.
* Always do a 1-epoch smoke test (`--set train.max_epochs=1 data.fraction=0.1`) before starting a 3-seed batch.

---

## 8. Timeline & milestones

Start date: Mon 23 Nov 2026. **Shift the dates to match the official CS464 deadlines** once they are announced.

| Milestone | Weeks | Dates | Exit criteria |
|-----------|-------|-------|---------------|
| **M0 – Foundations** | W1 | 23 Nov – 29 Nov | Split committed, contracts implemented with tests: Trainer, metrics/results I/O, model registry, HOG features, escnn model |
| **M1 – Pipelines & baselines** | W2 | 30 Nov – 6 Dec | Dataset/GPU preprocessing merged, Kaggle runner works end to end, statmorph features cached, pretrained backbones smoke-tested, first escnn run |
| **M2 – Core experiments** | W3–W4 | 7 Dec – 20 Dec | Classical baselines, tuned deep models, family comparison, E1a, E3, E4 runs complete |
| **M3 – Analysis** | W5 | 21 Dec – 27 Dec | E1b sweep + headline figure, E2, E5, E6, classical orientation/data-efficiency, stretch SDSS |
| **M4 – Report** | W6 | 28 Dec – 3 Jan | Final report + slides, all figures regenerated from committed results |

Critical path: **#3 split → #9 dataset → #4 trainer → #12 backbones → #15 tuning → #19 family → #23 synthesis → #27 report.**
Until #9 is merged, people can develop against the contracts using the synthetic fixtures in `tests/conftest.py`.

---

## 9. Risks & mitigations
| Risk | Impact | Mitigation |
|------|--------|-----------|
| statmorph is slow or fails on some galaxies | Delays #11 / #14 | joblib parallelism, per-galaxy timeout, cache results, impute + `morph_failed` flag. LR/SVM/XGB can start on HOG + color alone. |
| escnn is hard to install or train | E1 loses its key arm | Pin the version, test on Colab in week 1 (#8), keep the model small (128 px input). Fallback: C4 group with exact 90° rotations. |
| GPU quota runs out | Missing runs | Run registry in `docs/RUNS.md`, spread runs across accounts, short tuning schedules, reuse runs between experiments (§6). |
| Results not comparable | Invalid conclusions | One split, one preprocessing module, one evaluation function. Every config is stored inside its result JSON. |
| Teammate blocked by a dependency | Delays | Contracts + synthetic fixtures from day 1, and a weekly sync. Raise blockers in the issue within 24 h. |
| Test-set leakage | Inflated numbers | Test evaluation only through `evaluate()` after training ends. PR review checks this. |
| Duplicates leak across splits | Inflated numbers | Dedup before splitting (#3). A unit test checks that no RA/Dec pair within 1″ spans two splits. |

---

## 10. Definition of done (project level)
- [ ] `split_v1.csv` committed; every result file reports `split_version: v1`.
- [ ] Every row in every report table regenerates with `python scripts/aggregate_results.py`.
- [ ] Deep results report mean ± std over 3 seeds. Classical seed handling is stated.
- [ ] All six experiments + the family comparison have a table/figure and a written conclusion that answers RQ1–RQ6.
- [ ] The headline orientation figure includes **every** model family.
- [ ] CI is green on `main`. README explains how to reproduce one run end to end.
- [ ] Final report and slides are submitted.
