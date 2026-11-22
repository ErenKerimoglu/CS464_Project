"""Trainer, optimizer/scheduler and scripts/train.py (#4). Tiny SmallCNN on synthetic data, CPU only."""
from __future__ import annotations

import copy
import csv
import importlib.util
import json

import numpy as np
import pytest
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from galaxy.constants import REPO_ROOT
from galaxy.data.dataset import GalaxyDataset
from galaxy.data.transforms import GalaxyPreprocess
from galaxy.evaluation.predict import model_from_config
from galaxy.evaluation.results import load_eval, run_dir
from galaxy.training.optim import build_optimizer, build_scheduler, lr_factor, param_groups
from galaxy.training.trainer import Trainer

BAND_STATS = {"mean": [0.2, 0.2, 0.2], "std": [0.15, 0.15, 0.15], "crop": 180}
CFG = {
    "experiment": "debug", "run_name": "tiny", "seed": 0,
    "data": {"batch_size": 8, "num_workers": 0, "eval_batch_multiplier": 2},
    "preprocess": {"policy": "none", "out_size": 32, "crop": 180},
    "model": {"name": "small_cnn", "pretrained": False, "input_size": 32, "dropout": 0.0,
              "freeze_backbone": False},
    "train": {"max_epochs": 3, "lr": 3e-3, "head_lr": None, "weight_decay": 0.05, "warmup_epochs": 1,
              "grad_clip": 1.0, "amp": True, "early_stopping_patience": 8, "monitor": "val_macro_f1"},
    "loss": {"name": "ce", "label_smoothing": 0.1},
    "sampler": {"name": "shuffle"},
}


def make_cfg(**train):
    cfg = copy.deepcopy(CFG)
    cfg["train"].update(train)
    return cfg


@pytest.fixture
def datasets(synthetic_meta, synthetic_images):
    df = synthetic_meta.copy()
    pos = df.groupby("label").cumcount()
    df["split"] = np.select([pos < 4, pos == 4], ["train", "val"], "test")
    df["frac_rank"] = np.where(df.split == "train", pos / 4, np.nan)
    return {s: GalaxyDataset(s, images=synthetic_images, split_df=df) for s in ("train", "val", "test")}


def make_trainer(cfg, datasets, tmp_path):
    from galaxy.data.dataset import build_dataloaders

    torch.manual_seed(cfg["seed"])
    loaders = build_dataloaders(cfg, splits=("train", "val"), datasets=datasets)
    model = model_from_config(cfg)
    pre = GalaxyPreprocess.from_config(cfg, band_stats=BAND_STATS)
    return Trainer(cfg, model, loaders, pre, tmp_path / "run", tmp_path / "ckpt", device="cpu")


def read_log(path):
    with open(path, newline="") as f:
        return [{k: v for k, v in row.items() if k != "time"} for row in csv.DictReader(f)]


# ----------------------------------------------------------------------------- optim
def test_param_groups_no_decay_on_bias_and_norm():
    model = model_from_config(CFG)
    groups = {g["name"]: g for g in param_groups(model, make_cfg(head_lr=1e-2))}
    assert set(groups) == {"backbone_decay", "backbone_no_decay", "head_decay", "head_no_decay"}
    assert all(p.ndim > 1 for p in groups["backbone_decay"]["params"])
    assert all(p.ndim <= 1 for p in groups["backbone_no_decay"]["params"])
    assert groups["backbone_no_decay"]["weight_decay"] == 0 and groups["backbone_decay"]["weight_decay"] == 0.05
    assert groups["head_decay"]["lr"] == 1e-2 and groups["backbone_decay"]["lr"] == 3e-3
    n_opt = sum(p.numel() for g in groups.values() for p in g["params"])
    assert n_opt == sum(p.numel() for p in model.parameters())


def test_frozen_params_are_not_optimized():
    model = model_from_config(CFG)
    for p in model.backbone_parameters():
        p.requires_grad = False
    names = {g["name"] for g in param_groups(model, CFG)}
    assert names == {"head_decay", "head_no_decay"}


def test_warmup_then_cosine():
    assert lr_factor(0, 10, 100) == pytest.approx(0.1)
    assert lr_factor(9, 10, 100) == pytest.approx(1.0)
    assert lr_factor(10, 10, 100) == pytest.approx(1.0)
    assert lr_factor(55, 10, 100) == pytest.approx(0.5)
    assert lr_factor(100, 10, 100) == pytest.approx(0.0)
    assert lr_factor(100, 10, 100, min_ratio=0.1) == pytest.approx(0.1)
    model = nn.Linear(2, 2)
    opt = build_optimizer(model, make_cfg(max_epochs=2, warmup_epochs=1, lr=1.0))
    sched = build_scheduler(opt, make_cfg(max_epochs=2, warmup_epochs=1, lr=1.0), steps_per_epoch=4)
    lrs = []
    for _ in range(8):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[:4] == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert lrs[4] == pytest.approx(1.0) and lrs[-1] < 0.2 and (np.diff(lrs[3:]) <= 0).all()


# ----------------------------------------------------------------------------- trainer
def test_fit_writes_checkpoints_log_and_summary(datasets, tmp_path):
    trainer = make_trainer(make_cfg(max_epochs=4), datasets, tmp_path)
    summary = trainer.fit()
    log = read_log(tmp_path / "run" / "train_log.csv")
    assert [int(r["epoch"]) for r in log] == [0, 1, 2, 3]
    assert set(log[0]) == {"epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1", "lr"}
    assert float(log[-1]["train_loss"]) < float(log[0]["train_loss"]), "training loss should decrease"

    last = torch.load(tmp_path / "ckpt" / "last.pt", weights_only=False)
    best = torch.load(tmp_path / "ckpt" / "best.pt", weights_only=False)
    assert last["epoch"] == 3 and last["finished"]
    assert {"model", "optimizer", "scheduler", "scaler", "rng", "config"} <= set(last)
    assert set(best) == {"model", "config", "epoch", "monitor", "value"} and best["config"] == trainer.cfg
    assert best["value"] == pytest.approx(max(float(r["val_macro_f1"]) for r in log), abs=1e-6)

    on_disk = json.loads((tmp_path / "run" / "train_summary.json").read_text())
    assert on_disk["epochs_trained"] == summary["epochs_trained"] == 4
    assert on_disk["train_size"] == len(datasets["train"]) and sum(on_disk["train_class_counts"]) == 40
    assert on_disk["n_params"] > 0 and on_disk["peak_mem_gb"] is None  # CPU


def test_same_seed_same_metrics(datasets, tmp_path):
    make_trainer(make_cfg(max_epochs=2), datasets, tmp_path / "a").fit()
    make_trainer(make_cfg(max_epochs=2), datasets, tmp_path / "b").fit()
    assert read_log(tmp_path / "a" / "run" / "train_log.csv") == read_log(tmp_path / "b" / "run" / "train_log.csv")


def test_resume_continues_exactly(datasets, tmp_path):
    cfg = make_cfg(max_epochs=3)
    full = make_trainer(cfg, datasets, tmp_path / "full")
    full.fit()

    killed = make_trainer(cfg, datasets, tmp_path / "killed")
    original = killed.train_one_epoch

    def crash_in_epoch_1(epoch):
        if epoch == 1:
            raise KeyboardInterrupt("session ended")
        return original(epoch)

    killed.train_one_epoch = crash_in_epoch_1
    with pytest.raises(KeyboardInterrupt):
        killed.fit()
    assert torch.load(tmp_path / "killed" / "ckpt" / "last.pt", weights_only=False)["epoch"] == 0

    resumed = make_trainer(cfg, datasets, tmp_path / "killed")  # a fresh process: new model, new optimizer
    summary = resumed.fit(resume=True)
    assert summary["epochs_trained"] == 3
    assert read_log(tmp_path / "full" / "run" / "train_log.csv") == read_log(tmp_path / "killed" / "run" /
                                                                              "train_log.csv")
    for a, b in zip(full.model.state_dict().values(), resumed.model.state_dict().values(), strict=True):
        assert torch.equal(a, b)

    again = make_trainer(cfg, datasets, tmp_path / "killed")  # resuming a finished run trains nothing more
    assert again.fit(resume=True)["epochs_trained"] == 3
    assert len(read_log(tmp_path / "killed" / "run" / "train_log.csv")) == 3


def test_early_stopping(datasets, tmp_path):
    trainer = make_trainer(make_cfg(max_epochs=10, lr=0.0, early_stopping_patience=2), datasets, tmp_path)
    summary = trainer.fit()  # lr = 0: val macro-F1 never improves after epoch 0
    assert summary["stopped_early"] and summary["epochs_trained"] == 3 and summary["best_epoch"] == 0


def test_freeze_backbone_trains_only_head(datasets, tmp_path):
    cfg = make_cfg(max_epochs=1, weight_decay=0.0)
    cfg["model"]["freeze_backbone"] = True
    trainer = make_trainer(cfg, datasets, tmp_path)
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    trainer.fit()
    after = trainer.model.state_dict()
    for k in before:
        changed = not torch.equal(before[k], after[k])
        assert changed == k.startswith("fc."), k  # BN running stats included: backbone stays in eval mode
    assert trainer.summary()["n_trainable_params"] == sum(p.numel() for p in trainer.model.head_parameters())


def test_weighted_loss_and_oversampler_plug_in(datasets, tmp_path):
    from galaxy.data.dataset import build_dataloaders
    from galaxy.training.samplers import build_sampler

    cfg = make_cfg(max_epochs=1)
    cfg["loss"] = {"name": "focal", "gamma": 2.0}
    cfg["sampler"] = {"name": "oversample"}
    sampler = build_sampler(cfg, datasets["train"].labels)
    loaders = build_dataloaders(cfg, sampler=sampler, splits=("train", "val"), datasets=datasets)
    trainer = Trainer(cfg, model_from_config(cfg), loaders,
                      GalaxyPreprocess.from_config(cfg, band_stats=BAND_STATS), tmp_path / "r", tmp_path / "c",
                      device="cpu")
    assert trainer.fit()["epochs_trained"] == 1


# ----------------------------------------------------------------------------- scripts/train.py → evaluate.py
def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_script", REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_train_script_end_to_end_then_reevaluate(datasets, tmp_path, monkeypatch):
    import galaxy.constants

    cfg_file = tmp_path / "tiny.yaml"
    cfg_file.write_text(yaml.safe_dump({
        "defaults": ["models/small_cnn.yaml"], "experiment": "debug", "run_name": "tiny",
        "model": {"input_size": 32}, "preprocess": {"out_size": 32},
        "data": {"num_workers": 0, "batch_size": 8},
    }))
    train = _load_script("train")
    monkeypatch.setattr(galaxy.constants, "CKPT_DIR", tmp_path / "checkpoints")
    monkeypatch.setattr(train, "CKPT_DIR", tmp_path / "checkpoints")
    monkeypatch.setattr(train, "make_run_dir", lambda e, r, s: run_dir(e, r, s, root=tmp_path / "runs"))
    argv = ["--config", str(cfg_file), "--seed", "1", "--device", "cpu", "--set", "train.max_epochs=2"]

    summary = train.main(argv, datasets=datasets)
    rd = run_dir("debug", "tiny", 1, root=tmp_path / "runs")
    assert summary["epochs_trained"] == 2
    assert (tmp_path / "checkpoints" / "debug" / "tiny" / "seed1" / "best.pt").exists()
    test_eval = load_eval(rd / "eval_test.json")
    assert {p.name for p in rd.iterdir()} >= {"eval_val.json", "eval_test.json", "preds_test.npz",
                                              "train_log.csv", "train_summary.json"}
    assert test_eval["config"]["seed"] == 1 and test_eval["config"]["train"]["max_epochs"] == 2
    assert test_eval["model"] == "small_cnn" and test_eval["seed"] == 1
    assert test_eval["extra"]["n_params"] == summary["n_params"]
    assert test_eval["metrics"]["n_samples"] == len(datasets["test"])

    with pytest.raises(SystemExit, match="--resume"):  # never silently overwrite a run
        train.main(argv, datasets=datasets)

    # Acceptance (#5): re-evaluating the run reproduces the JSON, except the timestamp.
    evaluate = _load_script("evaluate")
    monkeypatch.setattr(evaluate, "build_dataloaders",
                        lambda c, splits: {s: DataLoader(datasets[s], batch_size=16) for s in splits})
    evaluate.main(["--run-dir", str(rd), "--eval", "test", "--device", "cpu"])
    again = load_eval(rd / "eval_test.json")
    test_eval.pop("timestamp"), again.pop("timestamp")
    assert again == test_eval
