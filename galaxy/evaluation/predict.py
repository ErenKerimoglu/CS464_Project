"""Model inference with optional raw-image transform hook, and the shared ``evaluate()``. Issue #5 (owner: Ege).

Every deep evaluation (``scripts/train.py`` after training, ``scripts/evaluate.py``, TTA #17, rotation sweep #23,
corruptions #22, SDSS #26) goes through :func:`predict`, so they all use the same preprocessing and inference code.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn

from galaxy.data.transforms import to_float01
from galaxy.evaluation.metrics import compute_metrics
from galaxy.evaluation.results import save_eval


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """``None``/``"auto"`` → CUDA if available, else CPU."""
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@torch.inference_mode()
def predict(model: nn.Module, loader, preprocess: nn.Module, device: str | torch.device | None = None,
            transform_fn: Callable | None = None, tta: Callable | None = None,
            amp: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(probs[N,C] float32, y_true[N] int64, orig_index[N] int64)`` in loader order.

    Args:
        loader: yields ``(uint8 images [B,3,256,256], labels, orig_index)`` (``build_dataloaders``).
        preprocess: ``GalaxyPreprocess``; always called with ``train=False`` (no augmentation).
        transform_fn: ``f(x_float01[B,3,256,256]) -> same shape`` applied **before** ``preprocess``
            (rotation sweep #23, corruptions #22).
        tta: ``f(x_preprocessed[B,3,S,S]) -> list of views``; the softmax outputs of all views are averaged (#17).
        amp: run the model under fp16 autocast (CUDA only). Off by default so evaluation is bit-reproducible.
    The model's train/eval mode is restored afterwards.
    """
    device = resolve_device(device)
    was_training = model.training
    model.eval()
    preprocess.to(device).eval()
    use_amp = amp and device.type == "cuda"
    probs, labels, indices = [], [], []
    try:
        for batch in loader:
            x, y, oi = batch[0], batch[1], batch[2]
            x = x.to(device, non_blocking=True)
            if transform_fn is not None:
                x = transform_fn(to_float01(x))
            x = preprocess(x, train=False)
            views = tta(x) if tta is not None else [x]
            p = None
            for view in views:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(view)
                pv = torch.softmax(logits.float(), dim=1)
                p = pv if p is None else p + pv
            probs.append((p / len(views)).cpu())
            labels.append(torch.as_tensor(y).cpu())
            indices.append(torch.as_tensor(oi).cpu())
    finally:
        model.train(was_training)
    if not probs:
        raise ValueError("the loader yielded no batches")
    return (torch.cat(probs).numpy().astype(np.float32), torch.cat(labels).numpy().astype(np.int64),
            torch.cat(indices).numpy().astype(np.int64))


def default_checkpoint(run_dir: Path, name: str = "best.pt") -> Path:
    """``checkpoints/<experiment>/<run_name>/seed<k>/<name>`` for ``results/runs/<experiment>/<run_name>/seed<k>``.

    The checkpoint root is ``galaxy.constants.CKPT_DIR`` (``$GALAXY_CKPT_DIR`` overrides it, e.g. on Drive).
    """
    from galaxy.constants import CKPT_DIR

    run_dir = Path(run_dir)
    return CKPT_DIR.joinpath(*run_dir.parts[-3:], name)


def model_from_config(config: dict, num_classes: int | None = None, load_pretrained: bool = False) -> nn.Module:
    """Build the model of ``config["model"]``. By default **without** downloading pretrained weights (evaluation
    loads a checkpoint anyway); training passes ``load_pretrained=True`` to honour ``model.pretrained``."""
    from galaxy.constants import NUM_CLASSES
    from galaxy.models import build_model

    kwargs = dict(config["model"])
    name = kwargs.pop("name")
    kwargs["pretrained"] = bool(kwargs.get("pretrained", False)) and load_pretrained
    return build_model(name, num_classes=num_classes or NUM_CLASSES, **kwargs)


def load_checkpoint_model(ckpt_path: Path, device: str | torch.device | None = None) -> tuple[nn.Module, dict]:
    """Load ``best.pt``/``last.pt`` (a dict with ``"model"``: state_dict and ``"config"``: resolved config, as
    written by the Trainer, #4) → ``(model on device in eval mode, config)``."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint {ckpt_path} not found (set --ckpt or $GALAXY_CKPT_DIR)")
    # Our own files: they also hold optimizer/RNG state, which ``weights_only=True`` cannot unpickle.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" not in ckpt or "config" not in ckpt:
        raise KeyError(f"{ckpt_path} must contain 'model' and 'config' entries, has {sorted(ckpt)}")
    config = ckpt["config"]
    model = model_from_config(config)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}  # torch.compile prefix
    model.load_state_dict(state)
    return model.to(resolve_device(device)).eval(), config


def evaluate(model: nn.Module, loader, preprocess: nn.Module, run_dir: Path, eval_name: str, config: dict,
             device: str | torch.device | None = None, transform_fn: Callable | None = None,
             tta: Callable | None = None, extra: dict | None = None, amp: bool = False) -> dict:
    """``predict`` → ``compute_metrics`` → ``save_eval``. Returns the metrics dict.

    This is the only function that should produce deep-model result files.
    """
    probs, y_true, orig_index = predict(model, loader, preprocess, device, transform_fn=transform_fn, tta=tta,
                                        amp=amp)
    metrics = compute_metrics(y_true, probs, num_classes=probs.shape[1])
    save_eval(run_dir, eval_name, metrics, probs, y_true, orig_index, config, extra=extra)
    return metrics
