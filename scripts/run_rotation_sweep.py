"""Evaluate trained runs at every test rotation angle, optionally with TTA-8 (E1b, issue #23).

    python scripts/run_rotation_sweep.py --runs "results/runs/prelim/resnet50_*/seed0"
    python scripts/run_rotation_sweep.py --runs "results/runs/model_family/*/seed*" --angles 0:360:15 --tta8
    python scripts/run_rotation_sweep.py --runs <run dirs> --summary-only --figure results/figures/e1/accuracy_vs_angle

For each run the model is loaded once, then ``eval_test_rot<DDD>[_tta8].json`` is written for every angle through the
shared ``evaluate()`` (so 0° reproduces the plain ``eval_test.json`` metrics exactly). Existing result files are
skipped unless ``--force``. At the end a per-angle table and a summary (mean / worst-case accuracy and orientation
sensitivity) are printed and saved to ``results/tables/``; ``--figure`` also draws accuracy vs. angle.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd
import torch

from galaxy.constants import TABLES_DIR
from galaxy.data.dataset import build_dataloaders
from galaxy.data.transforms import GalaxyPreprocess
from galaxy.evaluation.predict import default_checkpoint, evaluate, load_checkpoint_model, resolve_device
from galaxy.evaluation.results import eval_name_for, flatten_eval, load_eval, load_train_summary, training_extra
from galaxy.evaluation.rotation import (
    make_rotation_fn,
    parse_angles,
    plot_accuracy_vs_angle,
    summarize_sweep,
    sweep_table,
)
from galaxy.evaluation.tta import d4_views
from galaxy.models.utils import count_params
from galaxy.utils.config import apply_overrides
from galaxy.utils.seed import seed_everything

RUNTIME_KEYS = ("data.num_workers", "data.batch_size", "data.eval_batch_multiplier", "data.in_memory",
                "data.pin_memory")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True,
                   help="run dirs or glob patterns: results/runs/<experiment>/<run_name>/seed<k>")
    p.add_argument("--angles", default=None, help="start:stop:step or a,b,c (default: 0:360:15, 24 angles)")
    p.add_argument("--tta8", action="store_true", help="sweep with 8-view dihedral TTA (#17)")
    p.add_argument("--eval", choices=("val", "test"), default="test")
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true", help="fp16 inference (faster, not bit-reproducible)")
    p.add_argument("--force", action="store_true", help="re-evaluate angles whose result file already exists")
    p.add_argument("--summary-only", action="store_true", help="only aggregate existing result files")
    p.add_argument("--name", default="rotation_sweep", help="file stem for the tables in results/tables/")
    p.add_argument("--figure", type=Path, default=None, help="also save accuracy vs. angle here (.pdf + .png)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help=f"runtime data overrides, one of {', '.join(RUNTIME_KEYS)}")
    args = p.parse_args(argv)
    bad = [s for s in args.set if s.split("=", 1)[0] not in RUNTIME_KEYS]
    if bad:
        p.error(f"--set only accepts runtime keys {RUNTIME_KEYS}; got {bad}")
    return args


def expand_runs(patterns) -> list[Path]:
    runs = []
    for pat in patterns:
        hits = sorted(glob.glob(str(pat))) or ([pat] if Path(pat).is_dir() else [])
        runs += [Path(h) for h in hits if Path(h).is_dir()]
    return sorted(dict.fromkeys(runs))


def sweep_run(run_dir: Path, angles, split: str = "test", tta8: bool = False, device="auto", amp: bool = False,
              force: bool = False, overrides=(), loader=None) -> list[str]:
    """Evaluate one run at every angle. Returns the eval names that were (re)computed."""
    eval_names = {a: eval_name_for(split, a, tta8=tta8) for a in angles}
    todo = [a for a in angles if force or not (run_dir / f"eval_{eval_names[a]}.json").exists()]
    if not todo:
        print(f"{run_dir}: all {len(angles)} angles already evaluated (use --force to redo)")
        return []
    device = resolve_device(device)
    model, config = load_checkpoint_model(default_checkpoint(run_dir), device)
    preprocess = GalaxyPreprocess.from_config(config).to(device)
    if loader is None:
        loader = build_dataloaders(apply_overrides(config, list(overrides)), splits=(split,))[split]
    summary = load_train_summary(run_dir)
    base_extra = training_extra(summary)
    if base_extra.get("n_params") is None:
        base_extra["n_params"] = count_params(model)
    done = []
    for a in todo:
        extra = {**base_extra, "rotation_deg": float(a), "corruption": None, "severity": None,
                 "tta": "d4_8" if tta8 else None}
        m = evaluate(model, loader, preprocess, run_dir, eval_names[a], config, device=device,
                     transform_fn=make_rotation_fn(a), tta=d4_views if tta8 else None, extra=extra, amp=amp)
        print(f"  {run_dir.parent.name}/{run_dir.name} {eval_names[a]:>18}: acc {m['accuracy']:.4f} · "
              f"macro-F1 {m['macro_f1']:.4f}")
        done.append(eval_names[a])
    return done


def collect(runs, split: str = "test") -> pd.DataFrame:
    """All rotation result files of ``runs`` → flat DataFrame (``load_evals`` rows)."""
    rows = []
    for rd in runs:
        for path in sorted(Path(rd).glob(f"eval_{split}_rot*.json")):
            row = flatten_eval(load_eval(path))
            row["path"] = str(path)
            rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None, loader=None) -> int:
    args = parse_args(argv)
    runs = expand_runs(args.runs)
    if not runs:
        sys.exit(f"no run directories match {args.runs}")
    angles = parse_angles(args.angles)
    if not args.summary_only:
        seed_everything(0)
        torch.backends.cudnn.benchmark = False  # reproducible: re-running gives the same JSON
        torch.backends.cudnn.deterministic = True
        print(f"Sweeping {len(runs)} run(s) × {len(angles)} angles{' with TTA-8' if args.tta8 else ''}")
        for rd in runs:
            sweep_run(rd, angles, args.eval, args.tta8, args.device, args.amp, args.force, args.set, loader=loader)

    table = sweep_table(collect(runs, args.eval))
    if table.empty:
        sys.exit("no rotation results found")
    summary = summarize_sweep(table)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / f"{args.name}_by_angle.csv", index=False)
    summary.to_csv(TABLES_DIR / f"{args.name}_summary.csv", index=False)
    with pd.option_context("display.width", 160, "display.float_format", "{:.4f}".format):
        print(summary.to_string(index=False))
    print(f"tables → {TABLES_DIR / (args.name + '_by_angle.csv')}, {TABLES_DIR / (args.name + '_summary.csv')}")
    if args.figure:
        print("figure →", plot_accuracy_vs_angle(table, args.figure))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
