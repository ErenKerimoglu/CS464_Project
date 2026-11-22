"""Platform detection and path setup for the cloud runners (#10). Uses fake /kaggle and /content trees."""
from __future__ import annotations

import os

import numpy as np
import pytest

from galaxy.utils import env


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in (*env.ENV_KEYS, "GALAXY_PLATFORM", "GALAXY_DRIVE_DIR", "KAGGLE_KERNEL_RUN_TYPE", "COLAB_RELEASE_TAG"):
        monkeypatch.delenv(key, raising=False)
    yield


def _fake_cache(folder):
    folder.mkdir(parents=True)
    np.save(folder / "images_256.npy", np.zeros((2, 4, 4, 3), np.uint8))
    (folder / "meta.csv").write_text("orig_index\n0\n1\n")


def test_detect_platform(monkeypatch, tmp_path):
    monkeypatch.setattr(env, "KAGGLE_WORKING", tmp_path / "nope")
    monkeypatch.setattr(env.importlib.util, "find_spec", lambda name: None)
    assert env.detect_platform() == "local"
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release-colab_20261120")
    assert env.detect_platform() == "colab"
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    assert env.detect_platform() == "kaggle"
    monkeypatch.setenv("GALAXY_PLATFORM", "local")
    assert env.detect_platform() == "local"
    monkeypatch.setenv("GALAXY_PLATFORM", "aws")
    with pytest.raises(ValueError):
        env.detect_platform()


def test_find_kaggle_processed(tmp_path):
    assert env.find_kaggle_processed(tmp_path / "missing") is None
    (tmp_path / "other" / "processed").mkdir(parents=True)        # no images file: ignored
    assert env.find_kaggle_processed(tmp_path) is None
    target = tmp_path / "datasets" / "someone" / "galaxy10-decals-processed-v1" / "processed"
    _fake_cache(target)
    assert env.find_kaggle_processed(tmp_path) == target


def test_setup_env_kaggle(monkeypatch, tmp_path):
    monkeypatch.setattr(env, "KAGGLE_WORKING", tmp_path / "working")
    monkeypatch.setattr(env, "KAGGLE_INPUT", tmp_path / "input")
    _fake_cache(tmp_path / "input" / "galaxy10-decals-processed-v1" / "processed")
    out = env.setup_env("kaggle", verbose=False)
    assert out["platform"] == "kaggle"
    assert os.environ["GALAXY_CKPT_DIR"] == str(tmp_path / "working" / "checkpoints") == out["ckpt_dir"]
    assert os.environ["GALAXY_PROCESSED_DIR"] == str(tmp_path / "input" / "galaxy10-decals-processed-v1" /
                                                     "processed")
    assert os.environ["GALAXY_DATA_DIR"] == str(tmp_path / "working" / "data")


def test_existing_env_vars_win(monkeypatch, tmp_path):
    monkeypatch.setattr(env, "KAGGLE_WORKING", tmp_path / "working")
    monkeypatch.setattr(env, "KAGGLE_INPUT", tmp_path / "input")
    monkeypatch.setenv("GALAXY_CKPT_DIR", "/my/ckpts")
    out = env.setup_env("kaggle", verbose=False)
    assert out["ckpt_dir"] == os.environ["GALAXY_CKPT_DIR"] == "/my/ckpts"


def test_setup_env_colab_copies_drive_cache_to_local_disk(monkeypatch, tmp_path):
    monkeypatch.setattr(env, "COLAB_CONTENT", tmp_path / "content")
    drive = tmp_path / "content" / "drive" / "MyDrive" / "galaxy"
    _fake_cache(drive / "processed")
    out = env.setup_env("colab", drive_dir=drive, verbose=False)
    local = tmp_path / "content" / "data" / "processed"
    assert out["processed_dir"] == str(local) and (local / "images_256.npy").exists() and (local / "meta.csv").exists()
    assert out["ckpt_dir"] == str(drive / "checkpoints")
    assert not list(local.glob("*.part"))


def test_setup_env_colab_without_copy_reads_drive(monkeypatch, tmp_path):
    monkeypatch.setattr(env, "COLAB_CONTENT", tmp_path / "content")
    drive = tmp_path / "drive"
    _fake_cache(drive / "processed")
    monkeypatch.setenv("GALAXY_DRIVE_DIR", str(drive))
    out = env.setup_env("colab", copy_data_to_local=False, verbose=False)
    assert out["processed_dir"] == str(drive / "processed")


def test_setup_env_local_sets_nothing(monkeypatch):
    out = env.setup_env("local", verbose=False)
    assert not any(k in os.environ for k in env.ENV_KEYS)
    assert out["ckpt_dir"].endswith("checkpoints")


def test_describe_env_reports_missing_cache(tmp_path):
    text = env.describe_env({"platform": "local", "data_dir": str(tmp_path), "processed_dir": str(tmp_path),
                             "ckpt_dir": str(tmp_path)})
    assert "MISSING" in text and "platform       local" in text


def test_run_commands_streams_and_stops_on_failure(tmp_path, capsys, monkeypatch):
    import subprocess

    monkeypatch.setenv("GALAXY_TEST_VAR", "inherited")
    env.run_commands(["echo hello", "echo $GALAXY_TEST_VAR"], cwd=tmp_path)
    out = capsys.readouterr().out
    assert "hello" in out and "inherited" in out
    with pytest.raises(subprocess.CalledProcessError):
        env.run_commands(["exit 3", "echo never"], cwd=tmp_path)
    assert "never" not in capsys.readouterr().out


def test_export_results_copies_only_new_files(tmp_path):
    import time
    import zipfile

    repo = tmp_path / "repo"
    old = repo / "results" / "runs" / "model_family" / "resnet50" / "seed0" / "eval_test.json"
    old.parent.mkdir(parents=True)
    old.write_text("{}")
    os.utime(old, (0, 0))                         # committed earlier, untouched in this session
    start = time.time() - 1
    new_dir = repo / "results" / "runs" / "e3_data_eff" / "small_cnn_f010" / "seed1"
    new_dir.mkdir(parents=True)
    for name in ("eval_test.json", "preds_test.npz", "train_log.csv"):
        (new_dir / name).write_text("x")

    dest = tmp_path / "export"
    run_dirs = env.export_results(repo, dest, newer_than=start)
    assert [p.as_posix() for p in run_dirs] == ["e3_data_eff/small_cnn_f010/seed1"]
    assert (dest / "results/runs/e3_data_eff/small_cnn_f010/seed1/train_log.csv").exists()
    assert not (dest / "results/runs/model_family").exists()
    with zipfile.ZipFile(tmp_path / "export.zip") as z:
        assert "results/runs/e3_data_eff/small_cnn_f010/seed1/eval_test.json" in z.namelist()

    text = env.results_instructions(run_dirs, tmp_path / "export.zip", "abc1234")
    assert "git checkout -b results/e3_data_eff" in text and "git add results/runs/e3_data_eff" in text
    assert env.results_instructions([], "x.zip", "abc").startswith("No new result files")
