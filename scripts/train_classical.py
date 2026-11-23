"""Tune and evaluate classical models on cached galaxy features."""
from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import SVC

from galaxy.constants import CONFIGS_DIR, NUM_CLASSES, SEEDS, TABLES_DIR
from galaxy.evaluation.metrics import compute_metrics
from galaxy.evaluation.results import run_dir, save_eval
from galaxy.features.pipeline import (
    FEATURE_SET_NAMES,
    build_preprocessor,
    feature_columns_by_group,
    load_feature_matrix,
)

LOGGER = logging.getLogger("train_classical")
DEFAULT_PARAMS = {
    "logreg": {"C": 1.0, "class_weight": None},
    "svm_rbf": {"C": 1.0, "gamma": "scale", "class_weight": None},
    "xgboost": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1.0,
        "reg_lambda": 1.0,
    },
}
ABLATION_FEATURE_SETS = (("hog",), ("color",), ("morph",), FEATURE_SET_NAMES)
BEST_CONFIG_DIR = CONFIGS_DIR / "experiments" / "baselines_classical"
OPTUNA_DIR = Path(__file__).resolve().parents[1] / "results" / "optuna"
ABLATION_TABLE = TABLES_DIR / "baselines_classical_feature_ablation.csv"


def _load_xgboost():
    try:
        import xgboost
    except ImportError as exc:
        raise ImportError("XGBoost baselines require the optional dependency: pip install 'galaxy[classical]'") from exc
    return xgboost


def _xgboost_device(xgboost) -> str:
    build_info = getattr(xgboost, "build_info", None)
    if build_info is not None and build_info().get("USE_CUDA", False):
        LOGGER.info("Using XGBoost GPU histogram training")
        return "cuda"
    LOGGER.info("Using XGBoost CPU histogram training")
    return "cpu"


def _make_estimator(
    model_name: str,
    params: dict,
    seed: int,
    *,
    probabilities: bool,
):
    if model_name == "logreg":
        return LogisticRegression(
            C=float(params["C"]),
            class_weight=params["class_weight"],
            # lbfgs uses multinomial loss for this 10-class task across supported sklearn versions.
            solver="lbfgs",
            max_iter=5000,
            random_state=seed,
        )
    if model_name == "svm_rbf":
        return SVC(
            C=float(params["C"]),
            gamma=params["gamma"],
            class_weight=params["class_weight"],
            kernel="rbf",
            probability=probabilities,
            random_state=seed,
        )
    if model_name == "xgboost":
        xgboost = _load_xgboost()

        return xgboost.XGBClassifier(
            **params,
            objective="multi:softprob",
            num_class=NUM_CLASSES,
            eval_metric="mlogloss",
            tree_method="hist",
            device=_xgboost_device(xgboost),
            early_stopping_rounds=50,
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model: {model_name}")


def _fit_estimator(estimator, model_name: str, X_train, y_train, X_val, y_val) -> None:
    if model_name == "xgboost":
        estimator.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        estimator.fit(X_train, y_train)


def _suggest_params(trial, model_name: str) -> dict:
    if model_name == "logreg":
        return {
            "C": trial.suggest_float("C", 1e-4, 1e4, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
    if model_name == "svm_rbf":
        return {
            "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
            "gamma": trial.suggest_float("gamma", 1e-5, 1e1, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1200),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
        }
    raise ValueError(f"Unknown model: {model_name}")


def _tune(
    model_name: str,
    X_train,
    y_train: np.ndarray,
    X_val,
    y_val: np.ndarray,
    seed: int,
    feature_sets: Sequence[str],
    fraction: float,
    n_trials: int,
) -> dict:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Optuna tuning requires the optional dependency: pip install 'galaxy[tuning]'") from exc

    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    study_path = OPTUNA_DIR / f"classical_{model_name}.db"
    storage = f"sqlite:///{study_path.resolve().as_posix()}"
    feature_name = "_".join(feature_sets)
    study_name = f"classical_{model_name}_{feature_name}_fraction_{fraction:g}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    def objective(trial) -> float:
        params = _suggest_params(trial, model_name)
        estimator = _make_estimator(model_name, params, seed, probabilities=False)
        _fit_estimator(estimator, model_name, X_train, y_train, X_val, y_val)
        predictions = estimator.predict(X_val)
        return float(f1_score(
            y_val,
            predictions,
            labels=np.arange(NUM_CLASSES),
            average="macro",
            zero_division=0,
        ))

    study.optimize(objective, n_trials=n_trials)
    params = dict(study.best_params)
    config_path = BEST_CONFIG_DIR / f"{model_name}_best.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump({
            "model": model_name,
            "features": list(feature_sets),
            "fraction": fraction,
            "seed": seed,
            "params": params,
            "study_name": study_name,
            "best_value": float(study.best_value),
        }, sort_keys=False),
        encoding="utf-8",
    )
    LOGGER.info("Best validation macro-F1 %.4f; parameters saved to %s", study.best_value, config_path)
    return params


def _load_best_params(model_name: str, feature_sets: Sequence[str]) -> dict | None:
    config_path = BEST_CONFIG_DIR / f"{model_name}_best.yaml"
    if not config_path.is_file():
        return None
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if tuple(config.get("features", ())) != tuple(feature_sets):
        return None
    params = config.get("params")
    if not isinstance(params, dict):
        raise ValueError(f"Best-parameter config has no params mapping: {config_path}")
    return params


def _pca_summary(preprocessor) -> dict:
    if "hog" not in preprocessor.named_transformers_:
        return {"pca_components": None, "pca_explained_variance": None}
    pca = preprocessor.named_transformers_["hog"].named_steps["pca"]
    explained_variance = float(pca.explained_variance_ratio_.sum())
    LOGGER.info(
        "HOG PCA retained %d components (%.4f explained variance)",
        pca.n_components_,
        explained_variance,
    )
    return {
        "pca_components": int(pca.n_components_),
        "pca_explained_variance": explained_variance,
    }


def _validate_results_api() -> None:
    labels = np.array([0, 1], dtype=np.int64)
    probabilities = np.zeros((2, NUM_CLASSES), dtype=np.float64)
    probabilities[0, 0] = 1.0
    probabilities[1, 1] = 1.0
    try:
        metrics = compute_metrics(labels, probabilities)
        with TemporaryDirectory(prefix="galaxy_results_check_") as temporary_directory:
            output_dir = Path(temporary_directory)
            eval_path = save_eval(
                output_dir,
                "dependency_check",
                metrics,
                probabilities,
                labels,
                np.array([0, 1], dtype=np.int64),
                {"model": "classical_results_contract_check"},
                extra={},
            )
            if not Path(eval_path).is_file() or not (output_dir / "preds_dependency_check.npz").is_file():
                raise RuntimeError("save_eval did not create its JSON and NPZ result files")
    except NotImplementedError as exc:
        raise RuntimeError(
            "Classical training requires the metrics and save_eval implementation from issue #5"
        ) from exc


def _prepare_feature_data(feature_sets: Sequence[str], fraction: float) -> dict:
    X_train, y_train, train_indices = load_feature_matrix(
        "train", feature_sets=feature_sets, fraction=fraction,
    )
    X_val, y_val, val_indices = load_feature_matrix("val", feature_sets=feature_sets)
    X_test, y_test, test_indices = load_feature_matrix("test", feature_sets=feature_sets)

    preprocessor = build_preprocessor(feature_columns_by_group(X_train.columns))
    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)
    return {
        "train": (X_train, y_train, train_indices),
        "val": (X_val, y_val, val_indices),
        "test": (X_test, y_test, test_indices),
        "pca_summary": _pca_summary(preprocessor),
    }


def _run_one(
    model_name: str,
    feature_sets: Sequence[str],
    seed: int,
    fraction: float,
    params: dict,
    prepared: dict,
) -> dict:
    X_train, y_train, _ = prepared["train"]
    X_val, y_val, val_indices = prepared["val"]
    X_test, y_test, test_indices = prepared["test"]
    pca_summary = prepared["pca_summary"]

    estimator = _make_estimator(model_name, params, seed, probabilities=True)
    _fit_estimator(estimator, model_name, X_train, y_train, X_val, y_val)
    run_name = model_name if tuple(feature_sets) == FEATURE_SET_NAMES else f"{model_name}_{'_'.join(feature_sets)}"
    output_dir = run_dir("baselines_classical", run_name, seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": model_name,
        "features": list(feature_sets),
        "fraction": fraction,
        "seed": seed,
        "params": params,
        "preprocessing": {"hog_pca_variance": 0.95, **pca_summary},
    }
    evaluations = {}
    for eval_name, X_eval, y_eval, indices in (
        ("val", X_val, y_val, val_indices),
        ("test", X_test, y_test, test_indices),
    ):
        probabilities = estimator.predict_proba(X_eval)
        metrics = compute_metrics(y_eval, probabilities)
        save_eval(
            output_dir,
            eval_name,
            metrics,
            probabilities,
            y_eval,
            indices,
            config,
            extra={"features": list(feature_sets), **pca_summary},
        )
        evaluations[eval_name] = metrics

    return {
        "features": "all" if tuple(feature_sets) == FEATURE_SET_NAMES else "+".join(feature_sets),
        "seed": seed,
        "val_macro_f1": evaluations["val"]["macro_f1"],
        "test_macro_f1": evaluations["test"]["macro_f1"],
        **pca_summary,
    }


def _write_ablation_table(rows: list[dict]) -> Path:
    results = pd.DataFrame(rows)
    table = results.groupby("features", sort=False).agg(
        val_macro_f1_mean=("val_macro_f1", "mean"),
        val_macro_f1_std=("val_macro_f1", "std"),
        test_macro_f1_mean=("test_macro_f1", "mean"),
        test_macro_f1_std=("test_macro_f1", "std"),
        pca_components=("pca_components", "first"),
        pca_explained_variance=("pca_explained_variance", "first"),
    ).reset_index()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(ABLATION_TABLE, index=False)
    LOGGER.info("Feature-ablation results saved to %s", ABLATION_TABLE)
    return ABLATION_TABLE


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(DEFAULT_PARAMS), required=True)
    parser.add_argument("--tune", action="store_true", help="Tune hyperparameters on validation macro-F1")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--seed", type=int, help="Run one seed (default: 0 for LR/SVM, 0/1/2 for XGBoost)")
    parser.add_argument("--features", nargs="+", choices=FEATURE_SET_NAMES, default=None)
    parser.add_argument("--fraction", type=float, default=1.0, help="Nested train subset fraction in (0, 1]")
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run XGBoost with HOG, color, morphology, and all features; write one summary table",
    )
    args = parser.parse_args(argv)
    if args.n_trials < 1:
        parser.error("--n-trials must be positive")
    if not 0 < args.fraction <= 1:
        parser.error("--fraction must be in (0, 1]")
    if args.ablation and args.model != "xgboost":
        parser.error("--ablation requires --model xgboost")
    if args.ablation and args.features is not None:
        parser.error("--ablation selects its own feature sets; omit --features")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    _validate_results_api()
    feature_sets = tuple(args.features or FEATURE_SET_NAMES)
    tuning_features = FEATURE_SET_NAMES if args.ablation else feature_sets
    seeds = [args.seed] if args.seed is not None else (list(SEEDS) if args.model == "xgboost" else [0])
    tuning_seed = args.seed if args.seed is not None else 0
    prepared = None

    if args.tune:
        prepared = _prepare_feature_data(tuning_features, args.fraction)
        X_train_transformed, y_train, _ = prepared["train"]
        X_val_transformed, y_val, _ = prepared["val"]
        params = _tune(
            args.model,
            X_train_transformed,
            y_train,
            X_val_transformed,
            y_val,
            tuning_seed,
            tuning_features,
            args.fraction,
            args.n_trials,
        )
    else:
        params = _load_best_params(args.model, tuning_features) or DEFAULT_PARAMS[args.model]
        if args.ablation and _load_best_params(args.model, tuning_features) is None:
            raise FileNotFoundError(
                f"No tuned XGBoost parameters for {tuning_features}; run with --tune before --ablation"
            )

    if args.ablation:
        rows = []
        for feature_variant in ABLATION_FEATURE_SETS:
            variant_data = (
                prepared
                if prepared is not None and tuple(feature_variant) == tuple(tuning_features)
                else _prepare_feature_data(feature_variant, args.fraction)
            )
            for seed in seeds:
                rows.append(_run_one(
                    args.model, feature_variant, seed, args.fraction, params, variant_data,
                ))
        _write_ablation_table(rows)
    else:
        if prepared is None:
            prepared = _prepare_feature_data(feature_sets, args.fraction)
        for seed in seeds:
            _run_one(args.model, feature_sets, seed, args.fraction, params, prepared)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
