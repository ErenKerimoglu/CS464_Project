# data/

Only `data/splits/` is committed: `split_v1.csv`, `band_stats_v1.json` and `removed.csv` (#3). **Never regenerate them.**

Everything else is created locally or downloaded from the team's private Kaggle Dataset / Drive (#10):

| Path | Created by | Size |
|------|-----------|------|
| `raw/Galaxy10_DECals.h5` | `scripts/prepare_data.py --download-only` (#2) | about 2.5 GB |
| `processed/images_256.npy`, `processed/meta.csv` | `scripts/prepare_data.py` (#3) | about 3.5 GB (uint8 memmap) |
| `features/*_v1.parquet` | `scripts/extract_features.py` (#7, #11) | small |

Set `GALAXY_DATA_DIR` to use a different location (e.g. `/kaggle/input/...` or a Drive folder).
