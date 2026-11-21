"""Throughput benchmark of the input pipeline + augmentation figure. Issue #9 (owner: Eren).

    python scripts/benchmark_pipeline.py                         # rotflip → 224, memmap, 4 workers
    python scripts/benchmark_pipeline.py --in-memory --workers 2
    python scripts/benchmark_pipeline.py --figure results/figures/eda/augmentation_examples.png

Reports images/s for (1) the DataLoader alone, (2) DataLoader + GPU preprocessing (what training sees), and
(3) GPU preprocessing alone on a resident batch. Target on a Kaggle T4: (2) > 1500 img/s.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from galaxy.data.dataset import GalaxyDataset, build_dataloaders
from galaxy.data.transforms import GalaxyPreprocess


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def time_loader(loader, n_batches: int, device: str, preprocess=None, warmup: int = 5) -> float:
    it = iter(loader)
    for _ in range(warmup):
        x = next(it)[0]
    n, t0 = 0, None
    for i in range(n_batches):
        try:
            x = next(it)[0]
        except StopIteration:
            it = iter(loader)
            x = next(it)[0]
        if preprocess is not None:
            preprocess(x.to(device, non_blocking=True), train=True)
        if i == 0:
            _sync(device)
            t0 = time.perf_counter()
            continue
        n += len(x)
    _sync(device)
    return n / (time.perf_counter() - t0)


def time_preprocess(preprocess, batch: torch.Tensor, device: str, reps: int = 50) -> float:
    x = batch.to(device)
    for _ in range(5):
        preprocess(x, train=True)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        preprocess(x, train=True)
    _sync(device)
    return reps * len(x) / (time.perf_counter() - t0)


def augmentation_figure(ds: GalaxyDataset, out: Path, index: int = 0, n: int = 8, seed: int = 0) -> None:
    import matplotlib.pyplot as plt

    torch.manual_seed(seed)
    stats = {"mean": [0, 0, 0], "std": [1, 1, 1], "crop": 180}
    x = ds[index][0][None].repeat(n, 1, 1, 1)
    views = {
        "none": GalaxyPreprocess("none", 180, stats)(x[:1], train=True),
        "rotflip": GalaxyPreprocess("rotflip", 180, stats)(x, train=True),
        "d4": GalaxyPreprocess("d4", 180, stats)(x, train=True),
    }
    fig, axes = plt.subplots(3, n, figsize=(1.6 * n, 5.2))
    for r, (name, v) in enumerate(views.items()):
        for c in range(n):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(v):
                ax.imshow(v[c].permute(1, 2, 0).clamp(0, 1).numpy())
        axes[r, 0].set_title(name, loc="left", fontsize=9)
    fig.suptitle(f"Training augmentation policies (orig_index {ds.orig_index[index]}, 180 px crop)", fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[figure] {out}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", default="rotflip")
    p.add_argument("--out-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--in-memory", action="store_true")
    p.add_argument("--batches", type=int, default=100)
    p.add_argument("--figure", type=Path, default=None)
    a = p.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t = time.perf_counter()
    train = GalaxyDataset("train", in_memory=a.in_memory)
    load_s = time.perf_counter() - t
    cfg = {"data": {"batch_size": a.batch_size, "num_workers": a.workers}}
    loader = build_dataloaders(cfg, splits=("train",), datasets={"train": train})["train"]
    pre = GalaxyPreprocess(a.policy, a.out_size).to(device)

    res = {
        "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "policy": a.policy, "out_size": a.out_size, "batch_size": a.batch_size, "workers": a.workers,
        "in_memory": a.in_memory, "dataset_init_s": round(load_s, 1),
        "loader_img_s": round(time_loader(loader, a.batches, device)),
        "loader_plus_preprocess_img_s": round(time_loader(loader, a.batches, device, pre)),
        "preprocess_only_img_s": round(time_preprocess(pre, next(iter(loader))[0], device)),
    }
    print(json.dumps(res, indent=2))
    if a.figure:
        augmentation_figure(train, a.figure)


if __name__ == "__main__":
    main()
