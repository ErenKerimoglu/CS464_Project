"""Metrics and result-file contract (#5). Synthetic data only; no torch needed."""
from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn import metrics as skm

from galaxy.constants import NUM_CLASSES, SPLIT_VERSION
from galaxy.evaluation.metrics import METRIC_KEYS, compute_metrics, confusion_matrix, row_normalize
from galaxy.evaluation.results import (
    DEFAULT_EXTRA,
    eval_name_for,
    load_eval,
    load_evals,
    load_preds,
    parse_run_dir,
    run_dir,
    save_eval,
    split_from_eval_name,
)


# ----------------------------------------------------------------------------- metrics
def test_hand_computed_example():
    y_true = np.array([0, 0, 0, 1, 1, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    m = compute_metrics(y_true, y_pred, num_classes=3)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["per_class_recall"] == pytest.approx([2 / 3, 1 / 2, 1.0])
    assert m["per_class_precision"] == pytest.approx([1.0, 1 / 2, 1 / 2])
    f1 = [2 * 1 * (2 / 3) / (1 + 2 / 3), 0.5, 2 * 0.5 / 1.5]
    assert m["per_class_f1"] == pytest.approx(f1)
    assert m["macro_f1"] == pytest.approx(np.mean(f1))
    assert m["balanced_accuracy"] == pytest.approx(np.mean([2 / 3, 1 / 2, 1.0]))
    assert m["confusion_matrix"] == [[2, 1, 0], [0, 1, 1], [0, 0, 1]]
    assert m["n_samples"] == 6
    assert set(m) == set(METRIC_KEYS)


def _sklearn_reference(y_true, y_pred):
    labels = list(range(NUM_CLASSES))
    p, r, f, _ = skm.precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": skm.accuracy_score(y_true, y_pred),
        "macro_f1": skm.f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": skm.balanced_accuracy_score(y_true, y_pred),
        "per_class_recall": r, "per_class_precision": p, "per_class_f1": f,
        "confusion_matrix": skm.confusion_matrix(y_true, y_pred, labels=labels),
    }


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("n_true_classes", [10, 7, 2])
def test_matches_sklearn_on_random_inputs(seed, n_true_classes):
    """Includes subsets where classes are missing from y_true (and predictions of classes absent from y_true)."""
    rng = np.random.default_rng(seed)
    y_true = rng.choice(n_true_classes, size=200)
    probs = rng.dirichlet(np.ones(NUM_CLASSES), size=200)
    probs[np.arange(200), y_true] += rng.uniform(0, 0.6, 200)  # better than chance, still with errors
    y_pred = probs.argmax(1)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sklearn warns about predicted classes missing from y_true
        ref = _sklearn_reference(y_true, y_pred)
    for m in (compute_metrics(y_true, probs), compute_metrics(y_true, y_pred)):
        for key, value in ref.items():
            assert np.allclose(m[key], value), key


def test_missing_classes_keep_ten_entries():
    y_true = np.array([0, 0, 3, 3])
    m = compute_metrics(y_true, np.array([0, 3, 3, 3]))
    for key in ("per_class_recall", "per_class_precision", "per_class_f1"):
        assert len(m[key]) == NUM_CLASSES
    assert np.array(m["confusion_matrix"]).shape == (NUM_CLASSES, NUM_CLASSES)
    # macro-F1 averages only over the 2 classes that occur (sklearn's default), not over all 10
    assert m["macro_f1"] == pytest.approx(np.mean([2 / 3, 0.8]))
    assert m["balanced_accuracy"] == pytest.approx(np.mean([0.5, 1.0]))
    assert m["per_class_recall"][5] == 0.0


def test_metrics_are_json_serializable():
    m = compute_metrics(np.arange(10), np.arange(10))
    assert json.loads(json.dumps(m)) == m
    assert m["accuracy"] == m["macro_f1"] == 1.0


@pytest.mark.parametrize("y_true, y_pred, match", [
    (np.array([0, 1]), np.array([0]), "samples"),
    (np.array([], dtype=int), np.array([], dtype=int), "empty"),
    (np.array([0, 10]), np.array([0, 1]), "outside"),
    (np.array([0, 1]), np.zeros((2, 5)), "columns"),
])
def test_invalid_inputs(y_true, y_pred, match):
    with pytest.raises(ValueError, match=match):
        compute_metrics(y_true, y_pred)


def test_confusion_helpers():
    cm = confusion_matrix([0, 0, 1], [0, 1, 1], num_classes=3)
    assert cm.tolist() == [[1, 1, 0], [0, 1, 0], [0, 0, 0]]
    assert row_normalize(cm).tolist() == [[0.5, 0.5, 0], [0, 1, 0], [0, 0, 0]]


# ----------------------------------------------------------------------------- names
def test_eval_names():
    assert eval_name_for("test") == "test"
    assert eval_name_for("test", rotation_deg=45) == "test_rot045"
    assert eval_name_for("test", rotation_deg=360) == "test_rot000"
    assert eval_name_for("test", rotation_deg=7.5) == "test_rot007p5"
    assert eval_name_for("test", corruption="blur", severity=2) == "test_blur_s2"
    assert eval_name_for("val", tta8=True) == "val_tta8"
    assert eval_name_for("test", rotation_deg=90, tta8=True) == "test_rot090_tta8"
    with pytest.raises(ValueError):
        eval_name_for("test", corruption="blur")
    assert split_from_eval_name("test_rot045") == "test"
    assert split_from_eval_name("val") == "val"
    assert split_from_eval_name("sdss") == "sdss"


def test_run_dir_round_trip(tmp_path):
    d = run_dir("e3_data_eff", "resnet50_f010", 2, root=tmp_path)
    assert d == tmp_path / "e3_data_eff" / "resnet50_f010" / "seed2"
    assert parse_run_dir(d) == {"experiment": "e3_data_eff", "run_name": "resnet50_f010", "seed": 2}
    assert parse_run_dir(tmp_path / "somewhere") == {}


# ----------------------------------------------------------------------------- result files
def _fake_eval(rng, n=50):
    y = rng.integers(0, NUM_CLASSES, n)
    probs = rng.dirichlet(np.ones(NUM_CLASSES), size=n).astype(np.float32)
    return probs, y, rng.permutation(10_000)[:n]


DEEP_CONFIG = {"experiment": "model_family", "run_name": "resnet50", "seed": 1,
               "model": {"name": "resnet50", "pretrained": True}, "train": {"lr": 1e-4}}


def test_schema_round_trip(tmp_path, rng):
    probs, y, oi = _fake_eval(rng)
    metrics = compute_metrics(y, probs)
    d = run_dir("model_family", "resnet50", 1, root=tmp_path)
    path = save_eval(d, "test_rot045", metrics, probs, y, oi, DEEP_CONFIG, extra={"rotation_deg": 45})

    rec = load_eval(path)
    assert path == d / "eval_test_rot045.json"
    assert {k: rec[k] for k in ("experiment", "run_name", "seed", "model", "eval_name", "split")} == {
        "experiment": "model_family", "run_name": "resnet50", "seed": 1, "model": "resnet50",
        "eval_name": "test_rot045", "split": "test"}
    assert rec["split_version"] == SPLIT_VERSION
    assert rec["git_commit"] and rec["timestamp"]
    assert rec["config"] == DEEP_CONFIG
    assert rec["metrics"] == metrics
    assert rec["extra"] == {**DEFAULT_EXTRA, "rotation_deg": 45}

    preds = load_preds(d, "test_rot045")
    assert np.allclose(preds["probs"], probs, atol=1e-3)
    with np.load(d / "preds_test_rot045.npz") as z:
        assert z["probs"].dtype == np.float16
    assert np.array_equal(preds["y_true"], y) and np.array_equal(preds["orig_index"], oi)
    assert sorted(p.name for p in d.iterdir()) == ["eval_test_rot045.json", "preds_test_rot045.npz"]  # no temp files


def test_classical_style_config_without_torch_types(tmp_path, rng):
    """#14 passes ``model`` as a string, numpy values in config/extra, and no experiment/run_name keys."""
    probs, y, oi = _fake_eval(rng)
    config = {"model": "xgboost", "seed": np.int64(2), "params": {"eta": np.float32(0.1)}, "features": ("hog",)}
    d = run_dir("baselines_classical", "xgboost", 2, root=tmp_path)
    rec = load_eval(save_eval(d, "val", compute_metrics(y, probs), probs, y, oi, config,
                              extra={"pca_components": np.int64(37)}))
    assert (rec["experiment"], rec["run_name"], rec["seed"], rec["model"]) == ("baselines_classical", "xgboost", 2,
                                                                               "xgboost")
    assert rec["config"]["params"]["eta"] == pytest.approx(0.1)
    assert rec["config"]["features"] == ["hog"]
    assert rec["extra"]["pca_components"] == 37


def test_save_eval_outside_run_layout_uses_config(tmp_path, rng):
    probs, y, oi = _fake_eval(rng, n=2)
    rec = load_eval(save_eval(tmp_path, "dependency_check", compute_metrics(y, probs), probs, y, oi,
                              {"model": "check"}, extra={}))
    assert rec["experiment"] is None and rec["seed"] is None and rec["model"] == "check"
    assert rec["split"] == "dependency_check"


def test_save_eval_rejects_mismatched_shapes(tmp_path, rng):
    probs, y, oi = _fake_eval(rng)
    with pytest.raises(ValueError, match="shape"):
        save_eval(tmp_path, "test", compute_metrics(y, probs), probs[:-1], y, oi, {})
    with pytest.raises(ValueError, match="n_samples"):
        save_eval(tmp_path, "test", compute_metrics(y[:-1], probs[:-1]), probs, y, oi, {})


def test_resaving_is_identical_except_timestamp(tmp_path, rng):
    probs, y, oi = _fake_eval(rng)
    d = run_dir("model_family", "resnet50", 1, root=tmp_path)
    first = load_eval(save_eval(d, "test", compute_metrics(y, probs), probs, y, oi, DEEP_CONFIG))
    second = load_eval(save_eval(d, "test", compute_metrics(y, probs), probs, y, oi, DEEP_CONFIG))
    first.pop("timestamp"), second.pop("timestamp")
    assert first == second


def test_load_evals_flattens(tmp_path, rng):
    for seed in (0, 1):
        for eval_name in ("val", "test"):
            probs, y, oi = _fake_eval(rng)
            save_eval(run_dir("model_family", "resnet50", seed, root=tmp_path), eval_name,
                      compute_metrics(y, probs), probs, y, oi, {**DEEP_CONFIG, "seed": seed},
                      extra={"n_params": 23_000_000})
    df = load_evals(root=tmp_path)
    assert len(df) == 4
    assert list(df[["eval_name", "seed"]].itertuples(index=False, name=None)) == [
        ("test", 0), ("test", 1), ("val", 0), ("val", 1)]
    for col in ("accuracy", "macro_f1", "balanced_accuracy", "recall_Cigar", "precision_Dist", "f1_EdgeB",
                "n_params", "config", "path", "git_commit"):
        assert col in df.columns, col
    assert (df["n_params"] == 23_000_000).all()
    assert len(load_evals("**/eval_val.json", root=tmp_path)) == 2


def test_load_evals_empty(tmp_path):
    df = load_evals(root=tmp_path)
    assert df.empty and "macro_f1" in df.columns
