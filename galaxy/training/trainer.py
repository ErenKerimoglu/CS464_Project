"""Trainer. Issue #4 (owner: Ege). Contract: PROJECT_PLAN §4.3, recipe §3.2.

AMP, AdamW + warm-up/cosine, early stopping on val macro-F1, ``last.pt``/``best.pt`` checkpoints, exact resume,
``train_log.csv``, ``freeze_backbone`` (linear probe), and loss/sampler plug-ins from ``losses``/``samplers``.

Files::

    <ckpt_dir>/last.pt        every epoch: model, optimizer, scheduler, scaler, epoch, early-stopping state, RNG, config
    <ckpt_dir>/best.pt        best epoch on ``train.monitor``: {"model", "config", "epoch", "monitor", "value"}
    <run_dir>/train_log.csv   epoch, train_loss, val_loss, val_acc, val_macro_f1, lr, time
    <run_dir>/train_summary.json

**Exact resume.** Every epoch starts by reseeding python/numpy/torch and the train loader's shuffling from
``(seed, epoch)``, and ``last.pt`` stores everything else that changes between epochs. A run that is killed and
resumed with ``fit(resume=True)`` therefore follows the same trajectory as an uninterrupted one (bit-identical on
CPU; on GPU up to cuDNN non-determinism).

**The test set is never touched here.** Model selection and early stopping use the val loader only.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from galaxy.evaluation.metrics import compute_metrics
from galaxy.evaluation.predict import predict, resolve_device
from galaxy.evaluation.results import TRAIN_SUMMARY_FILE, to_jsonable
from galaxy.models.utils import count_params
from galaxy.training.losses import build_loss
from galaxy.training.optim import build_optimizer, build_scheduler

LOG_COLUMNS = ("epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1", "lr", "time")
MONITORS = ("val_macro_f1", "val_acc", "val_loss")


def _atomic_torch_save(obj, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # torch < 2.3
        return torch.cuda.amp.GradScaler(enabled=enabled)


def freeze_backbone(model: nn.Module) -> list[nn.Parameter]:
    """Linear probe: ``requires_grad=False`` on everything except ``model.head_parameters()``. Returns the head."""
    if not hasattr(model, "head_parameters"):
        raise AttributeError(f"{type(model).__name__} has no head_parameters(); cannot freeze its backbone")
    head = list(model.head_parameters())
    head_ids = {id(p) for p in head}
    for p in model.parameters():
        p.requires_grad = id(p) in head_ids
    return head


class Trainer:
    """Train one run. ``loaders`` needs ``"train"`` and ``"val"`` (yielding ``(uint8 images, labels, orig_index)``).

    Args:
        cfg: resolved config (``load_config``) including ``seed``.
        model: built model (pretrained weights already loaded).
        preprocess: ``GalaxyPreprocess``; called with ``train=True`` on train batches, ``False`` for validation.
        run_dir: ``results/runs/<experiment>/<run_name>/seed<k>`` (log + summary).
        ckpt_dir: ``checkpoints/<experiment>/<run_name>/seed<k>`` (``last.pt``, ``best.pt``).
        device: ``"cuda"``, ``"cpu"`` or ``None``/``"auto"``.
    """

    def __init__(self, cfg: dict, model: nn.Module, loaders: dict, preprocess: nn.Module, run_dir, ckpt_dir,
                 device: str | None = "cuda"):
        self.cfg = cfg
        self.t = cfg.get("train", {})
        self.device = resolve_device(device)
        self.seed = int(cfg.get("seed", 0))
        self.run_dir, self.ckpt_dir = Path(run_dir), Path(ckpt_dir)
        self.loaders = loaders
        self.model = model.to(self.device)
        self.preprocess = preprocess.to(self.device)

        self.frozen = bool(cfg.get("model", {}).get("freeze_backbone", False))
        self._head_ids: set[int] = set()
        if self.frozen:
            self._head_ids = {id(p) for p in freeze_backbone(self.model)}

        train_ds = loaders["train"].dataset
        self.class_counts = (train_ds.class_counts() if hasattr(train_ds, "class_counts")
                             else np.bincount(np.asarray(train_ds.labels), minlength=10))
        self.criterion = build_loss(cfg, self.class_counts).to(self.device)
        self.optimizer = build_optimizer(self.model, cfg)
        self.steps_per_epoch = len(loaders["train"])
        if self.steps_per_epoch == 0:
            raise ValueError("the train loader has no batches (dataset smaller than batch_size with drop_last?)")
        self.scheduler = build_scheduler(self.optimizer, cfg, self.steps_per_epoch)
        self.use_amp = bool(self.t.get("amp", True)) and self.device.type == "cuda"
        self.scaler = _make_grad_scaler(self.use_amp)

        self.max_epochs = int(self.t.get("max_epochs", 30))
        self.patience = self.t.get("early_stopping_patience")
        self.monitor = self.t.get("monitor", "val_macro_f1")
        if self.monitor not in MONITORS:
            raise ValueError(f"train.monitor must be one of {MONITORS}, got {self.monitor!r}")
        self.mode = "min" if self.monitor.endswith("loss") else "max"
        self.grad_clip = self.t.get("grad_clip")

        # state that last.pt carries across sessions
        self.epoch = -1               # last finished epoch (0-based)
        self.best_value = math.inf if self.mode == "min" else -math.inf
        self.best_epoch = -1
        self.epochs_no_improve = 0
        self.train_time_s = 0.0
        self.peak_mem_gb = 0.0
        self.finished = False
        self.stopped_early = False
        self.log_rows: list[dict] = []   # train_log.csv rows, also stored in last.pt (fresh clones can resume)

    # ------------------------------------------------------------------ helpers
    @property
    def last_path(self) -> Path:
        return self.ckpt_dir / "last.pt"

    @property
    def best_path(self) -> Path:
        return self.ckpt_dir / "best.pt"

    @property
    def log_path(self) -> Path:
        return self.run_dir / "train_log.csv"

    def _seed_epoch(self, epoch: int) -> None:
        """Reseed every RNG from (seed, epoch), so an epoch is reproducible whether or not the run was resumed."""
        s = (self.seed * 100_003 + epoch) % 2**32
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        loader = self.loaders["train"]
        for gen in (getattr(loader, "generator", None), getattr(getattr(loader, "sampler", None), "generator", None)):
            if isinstance(gen, torch.Generator):
                gen.manual_seed(s)

    def _set_train_mode(self) -> None:
        """Normal: ``model.train()``. Frozen backbone: the backbone runs in eval mode (BatchNorm statistics,
        dropout and drop-path frozen) and only modules made entirely of head parameters train."""
        if not self.frozen:
            self.model.train()
            return
        self.model.eval()
        for m in self.model.modules():
            params = list(m.parameters())
            if params and all(id(p) in self._head_ids for p in params):
                m.train()

    def _improved(self, value: float) -> bool:
        return value < self.best_value if self.mode == "min" else value > self.best_value

    def _log(self, msg: str) -> None:
        print(msg, flush=True)

    # ------------------------------------------------------------------ checkpoints
    def _state(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.epoch,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "epochs_no_improve": self.epochs_no_improve,
            "train_time_s": self.train_time_s,
            "peak_mem_gb": self.peak_mem_gb,
            "finished": self.finished,
            "stopped_early": self.stopped_early,
            "log": self.log_rows,
            "monitor": self.monitor,
            "config": self.cfg,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        }

    def save_last(self) -> None:
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        _atomic_torch_save(self._state(), self.last_path)

    def save_best(self) -> None:
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        _atomic_torch_save({"model": self.model.state_dict(), "config": self.cfg, "epoch": self.epoch,
                            "monitor": self.monitor, "value": self.best_value}, self.best_path)

    def load_last(self) -> None:
        """Restore everything from ``last.pt`` (exact resume)."""
        ckpt = torch.load(self.last_path, map_location="cpu", weights_only=False)
        old = {k: v for k, v in ckpt["config"].items() if k != "_meta"}
        new = {k: v for k, v in self.cfg.items() if k != "_meta"}
        if old != new:
            self._log("WARNING: the config differs from the one stored in last.pt; resuming with the new config.")
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        for key in ("epoch", "best_value", "best_epoch", "epochs_no_improve", "train_time_s", "peak_mem_gb",
                    "finished", "stopped_early"):
            setattr(self, key, ckpt[key])
        rng = ckpt["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        if ckpt.get("log") is not None:
            # Rewrite the log from the checkpoint: works in a fresh clone (Kaggle/Colab sessions) and drops rows of
            # an epoch that was logged but not checkpointed before a crash.
            self.log_rows = list(ckpt["log"])
            self._write_log(self.log_rows, mode="w")
        else:
            self._truncate_log(self.epoch)

    def _truncate_log(self, last_epoch: int) -> None:
        """Drop log rows of epochs after ``last_epoch`` (fallback for checkpoints without a stored log)."""
        if not self.log_path.exists():
            return
        with open(self.log_path, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if int(r["epoch"]) <= last_epoch]
        self.log_rows = rows
        self._write_log(rows, mode="w")

    def _write_log(self, rows: list[dict], mode: str = "a") -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        new_file = mode == "w" or not self.log_path.exists()
        with open(self.log_path, mode if not new_file else "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            if new_file:
                w.writeheader()
            w.writerows(rows)

    # ------------------------------------------------------------------ loop
    def train_one_epoch(self, epoch: int) -> float:
        """One pass over the train loader; returns the mean train loss."""
        self._set_train_mode()
        params = [p for g in self.optimizer.param_groups for p in g["params"]]
        total, n = 0.0, 0
        for x, y, _ in self.loaders["train"]:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            x = self.preprocess(x, train=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                logits = self.model(x)
            loss = self.criterion(logits.float(), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss ({loss.item()}) at epoch {epoch}; "
                                         "lower train.lr or check the data")
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if self.grad_clip:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(params, float(self.grad_clip))
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            total += loss.item() * len(y)
            n += len(y)
        return total / max(n, 1)

    def validate(self) -> dict:
        """val loss (plain cross-entropy, no smoothing/weights, so it is comparable across losses), acc, macro-F1."""
        probs, y, _ = predict(self.model, self.loaders["val"], self.preprocess, self.device, amp=self.use_amp)
        m = compute_metrics(y, probs, num_classes=probs.shape[1])
        nll = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None)).mean()
        return {"val_loss": float(nll), "val_acc": m["accuracy"], "val_macro_f1": m["macro_f1"]}

    def fit(self, resume: bool = False) -> dict:
        """Train until ``max_epochs`` or early stopping. Return a summary (best epoch, best val macro-F1, time)."""
        if resume and self.last_path.exists():
            self.load_last()
            self._log(f"Resumed from {self.last_path} after epoch {self.epoch + 1}/{self.max_epochs}")
        elif resume:
            self._log(f"--resume: no {self.last_path} yet, starting from scratch")
        if not (resume and self.last_path.exists()) and self.log_path.exists():
            self.log_path.unlink()  # fresh run: do not append to an old log

        self._log_header()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        while not self.finished and self.epoch + 1 < self.max_epochs:
            epoch = self.epoch + 1
            self._seed_epoch(epoch)
            t0 = time.time()
            train_loss = self.train_one_epoch(epoch)
            val = self.validate()
            dt = time.time() - t0

            self.epoch = epoch
            self.train_time_s += dt
            if self.device.type == "cuda":
                self.peak_mem_gb = max(self.peak_mem_gb, torch.cuda.max_memory_allocated(self.device) / 1e9)
            value = val[self.monitor]
            improved = self._improved(value)
            if improved:
                self.best_value, self.best_epoch, self.epochs_no_improve = value, epoch, 0
                self.save_best()
            else:
                self.epochs_no_improve += 1
            if self.patience and self.epochs_no_improve >= int(self.patience):
                self.finished = self.stopped_early = True
            if epoch + 1 >= self.max_epochs:
                self.finished = True

            lr = self.optimizer.param_groups[0]["lr"]
            row = {"epoch": epoch, "train_loss": f"{train_loss:.6f}", "val_loss": f"{val['val_loss']:.6f}",
                   "val_acc": f"{val['val_acc']:.6f}", "val_macro_f1": f"{val['val_macro_f1']:.6f}",
                   "lr": f"{lr:.3e}", "time": f"{dt:.1f}"}
            self.log_rows.append(row)
            self._write_log([row])
            self.save_last()
            self.write_summary()
            self._log(f"epoch {epoch + 1:>3}/{self.max_epochs}  train_loss {train_loss:.4f}  "
                      f"val_loss {val['val_loss']:.4f}  val_acc {val['val_acc']:.4f}  "
                      f"val_f1 {val['val_macro_f1']:.4f}  lr {lr:.2e}  {dt:.0f}s" + ("  *" if improved else ""))
            if self.stopped_early:
                self._log(f"Early stopping: no {self.monitor} improvement for {self.patience} epochs")

        summary = self.write_summary()
        self._log(f"Done. Best {self.monitor} {self.best_value:.4f} at epoch {self.best_epoch + 1}; "
                  f"train time {self.train_time_s / 60:.1f} min")
        return summary

    def _log_header(self) -> None:
        ds = self.loaders["train"].dataset
        counts = " ".join(str(int(c)) for c in self.class_counts)
        n_train, n_total = count_params(self.model, trainable_only=True), count_params(self.model)
        self._log(f"Run {self.cfg.get('experiment')}/{self.cfg.get('run_name')}/seed{self.seed} on {self.device}"
                  f" (AMP {'on' if self.use_amp else 'off'})")
        self._log(f"  train size {len(ds)} (fraction {getattr(ds, 'fraction', 1.0)}), class counts [{counts}]")
        self._log(f"  params {n_total:,} ({n_train:,} trainable{', backbone frozen' if self.frozen else ''}), "
                  f"{self.steps_per_epoch} steps/epoch, max {self.max_epochs} epochs")

    def summary(self) -> dict:
        return {
            "best_epoch": self.best_epoch,
            "monitor": self.monitor,
            "best_value": None if math.isinf(self.best_value) else self.best_value,
            "epochs_trained": self.epoch + 1,
            "max_epochs": self.max_epochs,
            "stopped_early": self.stopped_early,
            "finished": self.finished,
            "train_time_s": round(self.train_time_s, 1),
            "peak_mem_gb": round(self.peak_mem_gb, 3) if self.device.type == "cuda" else None,
            "n_params": count_params(self.model),
            "n_trainable_params": count_params(self.model, trainable_only=True),
            "train_size": len(self.loaders["train"].dataset),
            "train_class_counts": [int(c) for c in self.class_counts],
            "device": str(self.device),
        }

    def write_summary(self) -> dict:
        s = self.summary()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / TRAIN_SUMMARY_FILE).write_text(json.dumps(to_jsonable(s), indent=2) + "\n", encoding="utf-8")
        return s
