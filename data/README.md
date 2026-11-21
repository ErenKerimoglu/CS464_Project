# data/

Only `data/splits/` is committed (#3). **Never regenerate these files.** `prepare_data.py` refuses to overwrite them
and checks that a re-run reproduces them exactly.

| File | Content |
|------|---------|
| `split_v1.csv` | `orig_index,label,ra,dec,redshift,split,frac_rank` for every kept galaxy |
| `band_stats_v1.json` | train-only per-band mean/std (180 crop, [0,1] scale) |
| `removed_v1.csv` | images removed by deduplication: `orig_index,group_id,reason,label,kept_index` |
| `dedup_report_v1.json` | counts, radius sensitivity, split sizes, max class-share deviation |

Everything else is created locally or downloaded from the team's private Kaggle Dataset / Drive (#10):

| Path | Created by | Size |
|------|-----------|------|
| `raw/Galaxy10_DECals.h5` | `scripts/prepare_data.py --download-only` (#2) | about 2.5 GB |
| `processed/images_256.npy`, `processed/meta.csv` | `scripts/prepare_data.py` (#3) | about 3.5 GB (uint8 memmap) |
| `features/*_v1.parquet` | `scripts/extract_features.py` (#7, #11) | small |

Set `GALAXY_DATA_DIR` to use a different location (e.g. `/kaggle/input/...` or a Drive folder).
