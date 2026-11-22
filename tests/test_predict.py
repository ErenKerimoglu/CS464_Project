"""predict / evaluate / scripts/evaluate.py (#5) on synthetic data, CPU only."""
from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from galaxy.constants import NUM_CLASSES, REPO_ROOT
from galaxy.data.dataset import GalaxyDataset
from galaxy.data.transforms import GalaxyPreprocess
from galaxy.evaluation.predict import default_checkpoint, evaluate, load_checkpoint_model, predict
from galaxy.evaluation.results import load_eval, load_preds, run_dir
from galaxy.models import build_model

BAND_STATS = {"mean": [0.2, 0.2, 0.2], "std": [0.15, 0.15, 0.15], "crop": 180}


@pytest.fixture
def split_df(synthetic_meta):
    df = synthetic_meta.copy()
    pos = df.groupby("label").cumcount()
    df["split"] = np.select([pos < 4, pos == 4], ["train", "val"], "test")
    df["frac_rank"] = np.where(df.split == "train", pos / 4, np.nan)
    return df


@pytest.fixture
def loader(split_df, synthetic_images):
    ds = GalaxyDataset("test", images=synthetic_images, split_df=split_df)
    return DataLoader(ds, batch_size=4, shuffle=False)


@pytest.fixture
def preprocess():
    return GalaxyPreprocess(policy="rotflip", out_size=32, band_stats=BAND_STATS)


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(8, NUM_CLASSES)

    def forward(self, x):
        return self.fc(self.drop(self.conv(x).relu().mean((2, 3))))


def test_predict_shapes_order_and_determinism(loader, preprocess):
    torch.manual_seed(0)
    model = TinyNet().train()
    probs, y, oi = predict(model, loader, preprocess, device="cpu")
    ds = loader.dataset
    assert probs.shape == (len(ds), NUM_CLASSES) and probs.dtype == np.float32
    assert np.allclose(probs.sum(1), 1, atol=1e-5)
    assert np.array_equal(y, ds.labels) and np.array_equal(oi, ds.orig_index)
    assert model.training, "predict must restore the model's mode"
    probs2, _, _ = predict(model, loader, preprocess, device="cpu")  # eval mode: dropout and augmentation off
    assert np.array_equal(probs, probs2)


def test_transform_fn_receives_float01_raw_images(loader, preprocess):
    seen = []

    def transform_fn(x):
        seen.append((x.dtype, x.shape[1:], float(x.min()), float(x.max())))
        return torch.zeros_like(x)

    probs, _, _ = predict(TinyNet(), loader, preprocess, device="cpu", transform_fn=transform_fn)
    dtype, shape, lo, hi = seen[0]
    assert dtype == torch.float32 and tuple(shape) == (3, 256, 256) and 0 <= lo <= hi <= 1
    assert np.allclose(probs, probs[0])  # every image was replaced by the same black image


def test_tta_averages_softmax_over_views(loader, preprocess):
    torch.manual_seed(0)
    model = TinyNet()
    plain, _, _ = predict(model, loader, preprocess, device="cpu")
    same, _, _ = predict(model, loader, preprocess, device="cpu", tta=lambda x: [x, x, x])
    assert np.allclose(plain, same, atol=1e-6)
    flipped, _, _ = predict(model, loader, preprocess, device="cpu", tta=lambda x: [x.flip(3)])
    both, _, _ = predict(model, loader, preprocess, device="cpu", tta=lambda x: [x, x.flip(3)])
    assert np.allclose(both, (plain + flipped) / 2, atol=1e-6)


def test_evaluate_writes_result_files(tmp_path, loader, preprocess):
    d = run_dir("debug", "tiny", 0, root=tmp_path)
    cfg = {"experiment": "debug", "run_name": "tiny", "seed": 0, "model": {"name": "tiny"}}
    metrics = evaluate(TinyNet(), loader, preprocess, d, "test", cfg, device="cpu", extra={"n_params": 1})
    rec = load_eval(d / "eval_test.json")
    assert rec["metrics"] == metrics and rec["model"] == "tiny" and rec["extra"]["n_params"] == 1
    assert metrics["n_samples"] == len(loader.dataset)
    assert np.array_equal(load_preds(d, "test")["y_true"], loader.dataset.labels)


# ----------------------------------------------------------------------------- scripts/evaluate.py
def _load_script():
    spec = importlib.util.spec_from_file_location("evaluate_script", REPO_ROOT / "scripts" / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def trained_run(tmp_path, monkeypatch):
    """A fake finished run: best.pt in the Trainer's format (#4) + train_summary.json."""
    import galaxy.constants

    monkeypatch.setattr(galaxy.constants, "CKPT_DIR", tmp_path / "checkpoints")
    cfg = {"experiment": "model_family", "run_name": "small_cnn", "seed": 0,
           "model": {"name": "small_cnn", "pretrained": False, "input_size": 32, "dropout": 0.3},
           "preprocess": {"policy": "rotflip", "out_size": 32, "crop": 180},
           "data": {"batch_size": 4, "num_workers": 0}}
    d = run_dir("model_family", "small_cnn", 0, root=tmp_path / "runs")
    ckpt = default_checkpoint(d)
    assert ckpt == tmp_path / "checkpoints" / "model_family" / "small_cnn" / "seed0" / "best.pt"
    ckpt.parent.mkdir(parents=True)
    torch.manual_seed(0)
    model = build_model("small_cnn", pretrained=False, input_size=32)
    torch.save({"model": model.state_dict(), "config": cfg, "epoch": 3}, ckpt)
    d.mkdir(parents=True)
    (d / "train_summary.json").write_text(json.dumps({"train_time_s": 12.5, "peak_mem_gb": 0.1}))
    return d, cfg, model


def test_load_checkpoint_model(trained_run):
    d, cfg, model = trained_run
    loaded, loaded_cfg = load_checkpoint_model(default_checkpoint(d), device="cpu")
    assert loaded_cfg == cfg and not loaded.training
    for a, b in zip(model.state_dict().values(), loaded.state_dict().values(), strict=True):
        assert torch.equal(a, b)


def test_evaluate_script_end_to_end(trained_run, split_df, synthetic_images, monkeypatch):
    d, cfg, _ = trained_run
    script = _load_script()
    datasets = {s: GalaxyDataset(s, images=synthetic_images, split_df=split_df) for s in ("val", "test")}
    monkeypatch.setattr(script, "build_dataloaders",
                        lambda c, splits: {s: DataLoader(datasets[s], batch_size=c["data"]["batch_size"])
                                           for s in splits})
    monkeypatch.setattr(script.GalaxyPreprocess, "from_config",
                        classmethod(lambda cls, c: cls(policy="rotflip", out_size=32, band_stats=BAND_STATS)))

    assert script.main(["--run-dir", str(d), "--eval", "test", "--device", "cpu", "--set", "data.batch_size=3"]) == 0
    first = load_eval(d / "eval_test.json")
    assert first["config"] == cfg, "runtime --set overrides must not leak into the stored config"
    assert (first["experiment"], first["run_name"], first["seed"], first["model"]) == (
        "model_family", "small_cnn", 0, "small_cnn")
    assert first["extra"]["train_time_s"] == 12.5 and first["extra"]["n_params"] > 0
    assert first["metrics"]["n_samples"] == len(datasets["test"])

    script.main(["--run-dir", str(d), "--eval", "test", "--device", "cpu"])  # re-evaluation is reproducible
    second = load_eval(d / "eval_test.json")
    first.pop("timestamp"), second.pop("timestamp")
    assert first == second


def test_evaluate_script_rejects_bad_arguments(trained_run):
    d, _, _ = trained_run
    script = _load_script()
    with pytest.raises(SystemExit):
        script.parse_args(["--run-dir", str(d), "--corruption", "blur"])
    with pytest.raises(SystemExit):
        script.parse_args(["--run-dir", str(d), "--set", "train.lr=1"])


def test_unimplemented_variants_exit_cleanly(trained_run):
    """--rotate / --corruption depend on #23 / #22; until they land the script exits with a pointer."""
    d, _, _ = trained_run
    script = _load_script()
    from galaxy.evaluation import rotation

    try:
        rotation.make_rotation_fn(45)
    except NotImplementedError:
        with pytest.raises(SystemExit, match="not implemented"):
            script.main(["--run-dir", str(d), "--rotate", "45", "--device", "cpu"])
    else:
        pytest.skip("rotation sweep (#23) is implemented")
