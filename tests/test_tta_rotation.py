"""TTA-8 (#17) and the test-time rotation sweep (#23). CPU only, synthetic data."""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from galaxy.constants import REPO_ROOT
from galaxy.data.dataset import GalaxyDataset
from galaxy.data.transforms import GalaxyPreprocess, center_crop, rotate_batch
from galaxy.evaluation.predict import evaluate, model_from_config, predict
from galaxy.evaluation.results import load_eval, run_dir
from galaxy.evaluation.rotation import make_rotation_fn, parse_angles, summarize_sweep, sweep_table
from galaxy.evaluation.tta import d4_views

CFG = {
    "experiment": "debug", "run_name": "tiny", "seed": 0,
    "data": {"batch_size": 8, "num_workers": 0, "eval_batch_multiplier": 2},
    "preprocess": {"policy": "none", "out_size": 32, "crop": 180},
    "model": {"name": "small_cnn", "pretrained": False, "input_size": 32, "dropout": 0.0},
}


def _d4(x, k, flip):
    r = torch.rot90(x, k, dims=(2, 3))
    return torch.flip(r, dims=(3,)) if flip else r


def _tiny_net():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 10, 3, padding=1),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(10, 10)).eval()


# ----------------------------------------------------------------------------- TTA (#17)
def test_d4_views_are_eight_distinct_views():
    x = torch.rand(2, 3, 16, 16)
    views = d4_views(x)
    assert len(views) == 8 and all(v.shape == x.shape for v in views)
    flat = torch.stack([v.flatten() for v in views])
    assert torch.cdist(flat, flat).add(torch.eye(8)).min() > 0  # pairwise distinct
    with pytest.raises(ValueError):
        d4_views(torch.rand(2, 3, 16, 12))


@pytest.mark.parametrize("k,flip", [(k, f) for k in range(4) for f in (False, True)])
def test_tta_prediction_is_invariant_to_d4(k, flip):
    net, x = _tiny_net(), torch.rand(4, 3, 16, 16)

    def tta_probs(inp):
        with torch.no_grad():
            return torch.stack([net(v).softmax(1) for v in d4_views(inp)]).mean(0)

    assert torch.allclose(tta_probs(x), tta_probs(_d4(x, k, flip)), atol=1e-6)
    if k or flip:  # sanity: without TTA the network is not invariant
        with torch.no_grad():
            assert not torch.allclose(net(x).softmax(1), net(_d4(x, k, flip)).softmax(1), atol=1e-4)


# ----------------------------------------------------------------------------- rotation (#23)
def test_make_rotation_fn_geometry():
    x = torch.rand(2, 3, 256, 256)
    assert make_rotation_fn(0)(x) is x and make_rotation_fn(360)(x) is x
    assert torch.equal(make_rotation_fn(90)(x), torch.rot90(x, 1, dims=(2, 3)))       # exact, CCW
    assert torch.equal(make_rotation_fn(-90)(x), torch.rot90(x, 3, dims=(2, 3)))
    y = make_rotation_fn(37)(x)
    assert y.shape == x.shape
    # rotate full image then crop == training's fused rotate+crop (same sampling points)
    fused = rotate_batch(x, 37.0, crop=180, exact_if_possible=False)
    assert torch.allclose(center_crop(y, 180), fused, atol=1e-4)  # float rounding of affine_grid


@pytest.mark.parametrize("angle", [15, 45, 137])
def test_rotation_never_shows_padding_inside_the_crop(angle):
    ones = torch.ones(1, 3, 256, 256)
    assert center_crop(make_rotation_fn(angle)(ones), 180).min() > 0.999


def test_parse_angles():
    assert parse_angles(None) == [float(a) for a in range(0, 360, 15)]
    assert parse_angles("0:360:90") == [0, 90, 180, 270]
    assert parse_angles("0,45, 90") == [0, 45, 90]
    with pytest.raises(ValueError):
        parse_angles("0:360")


def test_rotation_zero_reproduces_plain_predictions(datasets):
    model = model_from_config(CFG).eval()
    pre = GalaxyPreprocess.from_config(CFG)
    loader = DataLoader(datasets["test"], batch_size=16)
    plain = predict(model, loader, pre, "cpu")
    rot0 = predict(model, loader, pre, "cpu", transform_fn=make_rotation_fn(0))
    for a, b in zip(plain, rot0, strict=True):
        np.testing.assert_array_equal(a, b)


# ----------------------------------------------------------------------------- sweep script
@pytest.fixture
def datasets(synthetic_meta, synthetic_images):
    df = synthetic_meta.copy()
    pos = df.groupby("label").cumcount()
    df["split"] = np.select([pos < 4, pos == 4], ["train", "val"], "test")
    df["frac_rank"] = np.where(df.split == "train", pos / 4, np.nan)
    return {s: GalaxyDataset(s, images=synthetic_images, split_df=df) for s in ("train", "val", "test")}


def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_script", REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sweep_script_end_to_end(datasets, tmp_path, monkeypatch):
    import galaxy.constants

    torch.manual_seed(0)
    model = model_from_config(CFG)
    ckpt = tmp_path / "checkpoints" / "debug" / "tiny" / "seed0"
    ckpt.mkdir(parents=True)
    torch.save({"model": model.state_dict(), "config": CFG}, ckpt / "best.pt")
    monkeypatch.setattr(galaxy.constants, "CKPT_DIR", tmp_path / "checkpoints")
    rd = run_dir("debug", "tiny", 0, root=tmp_path / "runs")
    rd.mkdir(parents=True)
    loader = DataLoader(datasets["test"], batch_size=16)
    evaluate(model.eval(), loader, GalaxyPreprocess.from_config(CFG), rd, "test", CFG, device="cpu")

    sweep = _load_script("run_rotation_sweep")
    monkeypatch.setattr(sweep, "TABLES_DIR", tmp_path / "tables")
    fig = tmp_path / "fig" / "acc_vs_angle"
    args = ["--runs", str(tmp_path / "runs" / "debug" / "*" / "seed0"), "--angles", "0:360:90", "--device", "cpu",
            "--figure", str(fig)]
    assert sweep.main(args, loader=loader) == 0
    assert sweep.main(args + ["--tta8"], loader=loader) == 0

    plain, rot0 = load_eval(rd / "eval_test.json"), load_eval(rd / "eval_test_rot000.json")
    assert rot0["metrics"] == plain["metrics"]                       # acceptance (#23): 0° reproduces test
    assert rot0["extra"]["rotation_deg"] == 0 and rot0["extra"]["tta"] is None
    # TTA-8 makes 90° rotations exactly invariant (rot90 of the full image = rot90 of the crop)
    p0 = np.load(rd / "preds_test_rot000_tta8.npz")["probs"].astype(np.float32)
    for a in (90, 180, 270):
        pa = np.load(rd / f"preds_test_rot{a:03d}_tta8.npz")["probs"].astype(np.float32)
        np.testing.assert_allclose(pa, p0, atol=2e-3)
    assert (tmp_path / "tables" / "rotation_sweep_summary.csv").exists()
    assert fig.with_suffix(".pdf").exists() and fig.with_suffix(".png").exists()

    # existing files are skipped; summary-only aggregates them
    assert sweep.main(args + ["--summary-only"], loader=loader) == 0


def test_sweep_table_and_summary():
    import pandas as pd

    rows = [{"experiment": "e", "run_name": "r", "seed": s, "split": "test", "rotation_deg": a, "tta": None,
             "corruption": None, "accuracy": acc + 0.01 * s, "macro_f1": acc - 0.1}
            for s in (0, 1) for a, acc in ((0, 0.9), (45, 0.7), (90, 0.9))]
    table = sweep_table(pd.DataFrame(rows))
    assert len(table) == 3 and (table["n_seeds"] == 2).all()
    summ = summarize_sweep(table).iloc[0]
    assert summ["worst_angle"] == 45 and summ["acc_worst"] == pytest.approx(0.705)
    assert summ["acc_mean_over_angles"] == pytest.approx((0.905 + 0.705 + 0.905) / 3)
    assert summ["orientation_sensitivity"] > 0
