"""Train a deep model, then evaluate its best checkpoint on val and test (issue #4).

    python scripts/train.py --config configs/models/small_cnn.yaml --seed 0
    python scripts/train.py --config configs/models/small_cnn.yaml --seed 0 --set train.max_epochs=1 data.fraction=0.1
    python scripts/train.py --config <yaml> --seed 0 --resume          # continue from last.pt after a session ends

Writes ``checkpoints/<experiment>/<run_name>/seed<k>/{last,best}.pt`` (``$GALAXY_CKPT_DIR`` overrides the root) and
``results/runs/<experiment>/<run_name>/seed<k>/{train_log.csv, train_summary.json, eval_val.json, eval_test.json}``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from galaxy.constants import CKPT_DIR
from galaxy.data.dataset import GalaxyDataset, build_dataloaders
from galaxy.data.transforms import GalaxyPreprocess
from galaxy.evaluation.predict import evaluate, load_checkpoint_model, model_from_config, resolve_device
from galaxy.evaluation.results import run_dir as make_run_dir
from galaxy.evaluation.results import training_extra
from galaxy.training.samplers import build_sampler
from galaxy.training.trainer import Trainer
from galaxy.utils.config import load_config
from galaxy.utils.seed import seed_everything


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides, e.g. train.lr=3e-4")
    p.add_argument("--resume", action="store_true", help="continue from last.pt (no-op if it does not exist)")
    p.add_argument("--no-test", action="store_true", help="evaluate on val only (e.g. while tuning)")
    p.add_argument("--device", default="auto")
    p.add_argument("--deterministic", action="store_true", help="deterministic cuDNN kernels (slower)")
    return p.parse_args(argv)


def run_paths(cfg: dict) -> tuple[Path, Path]:
    """``(results run dir, checkpoint dir)`` of a config with ``experiment``, ``run_name`` and ``seed``."""
    rd = make_run_dir(cfg["experiment"], cfg["run_name"], cfg["seed"])
    return rd, CKPT_DIR / cfg["experiment"] / cfg["run_name"] / f"seed{cfg['seed']}"


def build_run(cfg: dict, datasets: dict | None = None):
    """Model, preprocess and train/val loaders for ``cfg``. ``datasets`` lets tests inject synthetic datasets."""
    d = cfg.get("data", {})
    datasets = dict(datasets or {})
    train_ds = datasets.get("train") or GalaxyDataset("train", fraction=d.get("fraction", 1.0),
                                                      in_memory=d.get("in_memory", False))
    datasets["train"] = train_ds
    sampler = build_sampler(cfg, train_ds.labels)
    loaders = build_dataloaders(cfg, sampler=sampler, splits=("train", "val"), datasets=datasets)
    model = model_from_config(cfg, load_pretrained=True)
    preprocess = GalaxyPreprocess.from_config(cfg)
    in_size = getattr(model, "input_size", None)
    if in_size is not None and int(in_size) != preprocess.out_size:
        raise ValueError(f"preprocess.out_size={preprocess.out_size} but the model expects {in_size} px inputs; "
                         "set both in the config")
    return model, preprocess, loaders


def main(argv=None, datasets: dict | None = None) -> dict:
    args = parse_args(argv)
    cfg = load_config(args.config, args.set)
    cfg["seed"] = args.seed
    seed_everything(args.seed, deterministic=args.deterministic)
    if not args.deterministic:
        torch.backends.cudnn.benchmark = True
    device = resolve_device(args.device)
    run_dir, ckpt_dir = run_paths(cfg)
    if (ckpt_dir / "last.pt").exists() and not args.resume:
        sys.exit(f"{ckpt_dir / 'last.pt'} already exists. Pass --resume to continue it, or delete {ckpt_dir} "
                 "to start over.")

    model, preprocess, loaders = build_run(cfg, datasets)
    trainer = Trainer(cfg, model, loaders, preprocess, run_dir, ckpt_dir, device=device)
    summary = trainer.fit(resume=args.resume)
    best_path = trainer.best_path
    del trainer, model, loaders  # free GPU memory before reloading the best checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Final evaluation with the best checkpoint (val always; test only after training is over). It reloads best.pt
    # exactly like scripts/evaluate.py does, so re-evaluating the run later reproduces these files.
    best, _ = load_checkpoint_model(best_path, device)
    splits = [s for s in cfg.get("eval", {}).get("splits", ["val", "test"]) if not (args.no_test and s == "test")]
    eval_loaders = build_dataloaders(cfg, splits=tuple(splits), datasets=datasets)
    extra = training_extra(summary)
    for split in splits:
        m = evaluate(best, eval_loaders[split], preprocess, run_dir, split, cfg, device=device, extra=extra)
        print(f"{split}: accuracy {m['accuracy']:.4f} · macro-F1 {m['macro_f1']:.4f} → {run_dir}/eval_{split}.json")
    return summary


if __name__ == "__main__":
    main()
