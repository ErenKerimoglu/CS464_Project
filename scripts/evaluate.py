"""Evaluate a trained run and write ``eval_<name>.json`` + ``preds_<name>.npz`` into its run folder (issue #5).

    python scripts/evaluate.py --run-dir results/runs/model_family/resnet50/seed0 --eval test
    python scripts/evaluate.py --run-dir <run> --eval test --tta8                          # test_tta8      (#17)
    python scripts/evaluate.py --run-dir <run> --eval test --rotate 45                     # test_rot045    (#23)
    python scripts/evaluate.py --run-dir <run> --eval test --corruption blur --severity 2  # test_blur_s2   (#22)

The model and its config are loaded from ``checkpoints/<experiment>/<run_name>/seed<k>/best.pt`` (``--ckpt`` or
``$GALAXY_CKPT_DIR`` to override). ``--set`` only changes runtime data options for this evaluation (e.g.
``data.num_workers=0 data.batch_size=32``); the result file always stores the run's training config.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from galaxy.data.dataset import build_dataloaders
from galaxy.data.transforms import GalaxyPreprocess
from galaxy.evaluation.predict import default_checkpoint, evaluate, load_checkpoint_model, resolve_device
from galaxy.evaluation.results import TRAIN_SUMMARY_FILE, eval_name_for, load_train_summary, training_extra
from galaxy.models.utils import count_params
from galaxy.utils.config import apply_overrides
from galaxy.utils.seed import seed_everything

RUNTIME_KEYS = ("data.num_workers", "data.batch_size", "data.eval_batch_multiplier", "data.in_memory",
                "data.pin_memory")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True, help="results/runs/<experiment>/<run_name>/seed<k>")
    p.add_argument("--eval", choices=("val", "test"), default="test", help="which split to evaluate")
    p.add_argument("--ckpt", type=Path, default=None, help="checkpoint (default: best.pt of the run)")
    p.add_argument("--tta8", action="store_true", help="8-view dihedral test-time augmentation (#17)")
    p.add_argument("--rotate", type=float, default=None, help="rotate every test image by DEG degrees (#23)")
    p.add_argument("--corruption", default=None, help="noise | blur | downsample (#22)")
    p.add_argument("--severity", type=int, default=None, help="1-based corruption severity (#22)")
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true", help="fp16 inference (faster, not bit-reproducible)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help=f"runtime data overrides, one of {', '.join(RUNTIME_KEYS)}")
    args = p.parse_args(argv)
    if (args.corruption is None) != (args.severity is None):
        p.error("--corruption and --severity must be given together")
    bad = [s for s in args.set if s.split("=", 1)[0] not in RUNTIME_KEYS]
    if bad:
        p.error(f"--set only accepts runtime keys {RUNTIME_KEYS}; got {bad}")
    return args


def build_hooks(args: argparse.Namespace):
    """``(transform_fn, tta, extra)`` for the requested evaluation variant."""
    transform_fns, tta = [], None
    extra = {"rotation_deg": args.rotate, "corruption": args.corruption, "severity": args.severity,
             "tta": "d4_8" if args.tta8 else None}
    if args.rotate is not None:
        from galaxy.evaluation.rotation import make_rotation_fn
        transform_fns.append(make_rotation_fn(args.rotate))
    if args.corruption is not None:
        from galaxy.evaluation.corruptions import make_corruption_fn
        transform_fns.append(make_corruption_fn(args.corruption, args.severity))
    if args.tta8:
        from galaxy.evaluation.tta import d4_views
        tta = d4_views

    def transform_fn(x):  # rotation first (same geometry as the sweep), then the corruption
        for fn in transform_fns:
            x = fn(x)
        return x

    return (transform_fn if transform_fns else None), tta, extra


def main(argv=None) -> int:
    args = parse_args(argv)
    seed_everything(0)
    torch.backends.cudnn.benchmark = False  # reproducible: re-evaluating a run gives the same JSON
    torch.backends.cudnn.deterministic = True
    device = resolve_device(args.device)

    try:
        transform_fn, tta, extra = build_hooks(args)
    except NotImplementedError as e:
        sys.exit(f"This evaluation variant is not implemented yet: {e}")

    ckpt = args.ckpt or default_checkpoint(args.run_dir)
    model, config = load_checkpoint_model(ckpt, device)
    runtime_cfg = apply_overrides(config, args.set)
    preprocess = GalaxyPreprocess.from_config(config).to(device)
    loader = build_dataloaders(runtime_cfg, splits=(args.eval,))[args.eval]

    train_summary = load_train_summary(args.run_dir)
    extra = {**training_extra(train_summary), **extra}
    if extra["n_params"] is None:
        extra["n_params"] = count_params(model)
    eval_name = eval_name_for(args.eval, args.rotate, args.corruption, args.severity, args.tta8)
    print(f"Evaluating {ckpt} on {args.eval} ({len(loader.dataset)} images) as '{eval_name}' on {device}")
    if not train_summary:
        print(f"  note: no {TRAIN_SUMMARY_FILE} in {args.run_dir}; train time / peak memory stay null")
    try:
        metrics = evaluate(model, loader, preprocess, args.run_dir, eval_name, config, device=device,
                           transform_fn=transform_fn, tta=tta, extra=extra, amp=args.amp)
    except NotImplementedError as e:
        sys.exit(f"This evaluation variant is not implemented yet: {e}")
    print(f"  accuracy {metrics['accuracy']:.4f} · macro-F1 {metrics['macro_f1']:.4f} · "
          f"balanced acc {metrics['balanced_accuracy']:.4f} → {args.run_dir}/eval_{eval_name}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
