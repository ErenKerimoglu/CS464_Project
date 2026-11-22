# Team Kaggle Dataset: `galaxy10-decals-processed-v1`

A private Kaggle Dataset with the processed images, so nobody has to re-download 2.7 GB and rebuild the 3.5 GB
cache (about 12 minutes) at the start of every session. Part of #10.

| File | Content |
|------|---------|
| `processed/images_256.npy` | uint8 (17736, 256, 256, 3), row *i* = `orig_index` *i* |
| `processed/meta.csv` | split_v1 metadata |
| `splits/` | copies of `data/splits/*` from git (git stays the source of truth) |
| `README.md`, `SHA256SUMS.txt` | provenance (commit, date) and checksums |

## Use it (everyone)
1. In your Kaggle notebook, open the right-hand panel → **Input** → **+ Add Input**.
2. Search `galaxy10-decals-processed-v1` (under *Your Datasets* / *Shared with you*) → **Add**.
3. That's all. `galaxy.constants` finds `/kaggle/input/**/processed/images_256.npy` automatically, so
   `GalaxyDataset`, the benchmark and the training scripts read from it. `notebooks/01_pipeline_check.ipynb`
   skips the download when the dataset is attached.

The input folder is read-only. Feature caches, checkpoints and results are still written under `/kaggle/working`.
To point at another location, set `GALAXY_PROCESSED_DIR`.

## Colab: the Google Drive copy
Colab cannot attach Kaggle datasets, so `notebooks/colab_runner.ipynb` reads the same two files from Drive:
`<DRIVE_DIR>/processed/images_256.npy` and `meta.csv` (default `DRIVE_DIR` = `My Drive/CS464_galaxy`). They are
copied to the Colab disk at the start of each session, because memmap reads over Drive are slow.
Two ways to get them there:
* **Shared folder:** one member puts `processed/` into a Drive folder and shares it with the team. Everyone else
  opens it → **Organize → Add shortcut** → *My Drive*, named `CS464_galaxy`.
* **From Kaggle, per person:** in the Colab runner set `DATA_SOURCE = "kaggle"` and
  `KAGGLE_DATASET = "<owner>/galaxy10-decals-processed-v1"`, and add the Colab secrets `KAGGLE_USERNAME` /
  `KAGGLE_KEY`. With `SAVE_TO_DRIVE = True` the first run also writes the Drive copy.

## Build it (once, by one person; done by Eren)
1. kaggle.com → **Create → New Notebook** → **File → Import Notebook** → GitHub tab →
   `https://github.com/ErenKerimoglu/CS464_Project/blob/main/notebooks/02_build_kaggle_dataset.ipynb`
2. Right-hand panel → **Session options**: Accelerator **None** (CPU is enough), **Internet On**.
3. Top right **Save Version** → **Save & Run All (Commit)** → **Save**. It runs in the background (about 15 min),
   so you can close the tab.
4. When the version shows as finished: open the notebook's page (**Your Work → Code**) → **Output** tab →
   **New Dataset**. Title `galaxy10-decals-processed-v1`, visibility **Private** → **Create**.
5. On the dataset page → **⋮ / Share** (or *Settings → Sharing*) → add each teammate's **Kaggle username** with
   *Can view*.

Kaggle's button labels change from time to time. If one is named differently, look for the same action nearby.

## Rebuilding
Only needed if `split_v1` changes (it should not). Rebuild as `…-v2` and keep v1 so old runs stay reproducible.
