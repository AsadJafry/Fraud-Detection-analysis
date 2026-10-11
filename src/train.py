"""Training XGBoost models with different class-imbalance strategies.

All models are fitted on the training split and early-stopped on the
validation split. The test split is never passed to anything in this module.

Strategies:

- ``"none"``: train on the data as it is.
- ``"scale_pos_weight"``: weight each fraud row by (#legit / #fraud).
- ``"smote"``: add synthetic fraud rows to the training set (SMOTE).
- ``"undersample"``: randomly drop legit rows from the training set.
"""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"

SEED = 42
STRATEGIES = ["none", "scale_pos_weight", "smote", "undersample"]
EARLY_STOPPING_ROUNDS = 50

# Starting hyperparameters, used for the strategy comparison.
DEFAULT_PARAMS = {
    "n_estimators": 1000,       # upper bound; early stopping picks the real number
    "learning_rate": 0.1,
    "max_depth": 6,
    "min_child_weight": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

# Search space for the random search.
SEARCH_SPACE = {
    "max_depth": randint(3, 11),               # 3 to 10
    "learning_rate": loguniform(0.03, 0.3),
    "min_child_weight": loguniform(1, 50),
    "subsample": uniform(0.6, 0.4),            # 0.6 to 1.0
    "colsample_bytree": uniform(0.5, 0.5),     # 0.5 to 1.0
}


def check_xgboost_version(min_major: int = 2) -> str:
    """Make sure xgboost is recent enough for the constructor-based API.

    From xgboost 2.0, ``early_stopping_rounds`` and ``eval_metric`` are
    constructor arguments (no longer ``fit`` arguments), and predictions use
    the best early-stopped iteration automatically.
    """
    version = xgboost.__version__
    if int(version.split(".")[0]) < min_major:
        raise RuntimeError(f"xgboost >= {min_major}.0 required, found {version}.")
    return version


def scale_pos_weight_value(y) -> float:
    """Return #negatives / #positives, XGBoost's recommended class weight."""
    y = np.asarray(y)
    n_pos = int(y.sum())
    if n_pos == 0:
        raise ValueError("No positive rows: cannot compute scale_pos_weight.")
    return (len(y) - n_pos) / n_pos


def resample(X: pd.DataFrame, y: pd.Series, strategy: str, ratio: float = 0.1,
             seed: int = SEED) -> tuple:
    """Resample the TRAINING set for the "smote" and "undersample" strategies.

    ``ratio`` is the target fraud:legit ratio after resampling (0.1 means one
    fraud row per ten legit rows). A full 1:1 balance would mean ~2.6M
    synthetic rows for SMOTE, or throwing away 99.8% of legit rows when
    undersampling, so a moderate ratio is used for both.

    SMOTE creates new fraud rows by interpolating between a fraud row and one
    of its nearest fraud neighbours. Distances are computed on standardised
    features (otherwise the balance columns, in the millions, would decide
    every neighbour), then mapped back to the original units. Interpolated
    rows can have fractional values for binary features; tree models cope
    with this, but it is one reason synthetic rows are not real transactions.

    Other strategies return the data unchanged. Never call this on
    validation or test data.
    """
    if strategy in ("none", "scale_pos_weight"):
        return X, y
    if strategy == "smote":
        from imblearn.over_sampling import SMOTE

        scaler = StandardScaler().fit(X)
        X_scaled, y_res = SMOTE(sampling_strategy=ratio, random_state=seed).fit_resample(
            scaler.transform(X), y
        )
        X_res = pd.DataFrame(scaler.inverse_transform(X_scaled), columns=X.columns)
        return X_res, pd.Series(np.asarray(y_res), name=y.name)
    if strategy == "undersample":
        from imblearn.under_sampling import RandomUnderSampler

        X_res, y_res = RandomUnderSampler(sampling_strategy=ratio, random_state=seed).fit_resample(X, y)
        return X_res, y_res
    raise ValueError(f"Unknown strategy {strategy!r}; choose from {STRATEGIES}.")


def make_xgb(params: dict = None, scale_pos_weight: float = 1.0) -> XGBClassifier:
    """Build an XGBClassifier with fixed settings plus ``params``.

    Fixed: hist tree method, PR-AUC ("aucpr") as the early-stopping metric,
    early stopping after 50 rounds without improvement, seed 42.
    """
    all_params = {**DEFAULT_PARAMS, **(params or {})}
    return XGBClassifier(
        **all_params,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        random_state=SEED,
        n_jobs=-1,
    )


def train_xgb(X_train, y_train, X_val, y_val, strategy: str = "scale_pos_weight",
              params: dict = None, ratio: float = 0.1, resampled: tuple = None):
    """Train one XGBoost model with the given imbalance strategy.

    The training set is resampled first if the strategy asks for it (pass
    ``resampled=(X, y)`` to reuse an already resampled set). The validation
    set is only used for early stopping and is never resampled.

    Returns the fitted model; ``model.best_iteration`` is the number of the
    best boosting round on validation PR-AUC.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; choose from {STRATEGIES}.")
    X_fit, y_fit = resampled if resampled is not None else resample(X_train, y_train, strategy, ratio)
    spw = scale_pos_weight_value(y_fit) if strategy == "scale_pos_weight" else 1.0
    model = make_xgb(params, scale_pos_weight=spw)
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    return model


def compare_strategies(X_train, y_train, X_val, y_val, strategies=STRATEGIES,
                       ratio: float = 0.1) -> tuple:
    """Train one model per imbalance strategy with the same default settings.

    Returns ``(results, models, val_scores)``: a DataFrame with one row per
    strategy (validation PR-AUC decides the winner; precision and recall at
    0.5 are for context), plus dicts of fitted models and validation scores.
    """
    from src.evaluate import evaluate_model

    rows, models, val_scores = [], {}, {}
    for strategy in strategies:
        t0 = time.time()
        X_fit, y_fit = resample(X_train, y_train, strategy, ratio)
        model = train_xgb(X_train, y_train, X_val, y_val, strategy=strategy,
                          ratio=ratio, resampled=(X_fit, y_fit))
        seconds = time.time() - t0
        scores = model.predict_proba(X_val)[:, 1]
        m = evaluate_model(y_val, scores, threshold=0.5)
        models[strategy], val_scores[strategy] = model, scores
        rows.append({
            "strategy": strategy,
            "train rows": int(len(y_fit)),
            "train fraud share": float(np.mean(y_fit)),
            "trees": int(model.best_iteration) + 1,
            "val PR-AUC": m["pr_auc"],
            "val ROC-AUC": m["roc_auc"],
            "precision @0.5": m["precision"],
            "recall @0.5": m["recall"],
            "false alarms @0.5": m["fp"],
            "seconds": round(seconds),
        })
        print(f"  {strategy:<17} val PR-AUC {m['pr_auc']:.4f}  ({seconds:.0f}s)")
    return pd.DataFrame(rows).set_index("strategy"), models, val_scores


def random_search(X_train, y_train, X_val, y_val, strategy: str, n_trials: int = 20,
                  time_budget_s: float = 1800, ratio: float = 0.1, seed: int = SEED) -> tuple:
    """Small random hyperparameter search scored on validation PR-AUC.

    Each trial trains on the training split and early-stops on the
    validation split, so the time order is kept: there are no shuffled
    cross-validation folds mixing past and future rows. (This is the same
    as a single predefined train/validation split.)

    The search stops after ``n_trials`` trials or once ``time_budget_s``
    seconds have passed (a trial already running is allowed to finish).

    Because validation is used both for early stopping and for picking the
    winner, the best validation score is slightly optimistic. The untouched
    test split gives the honest estimate later.

    Returns ``(best_model, results)`` where ``results`` is a DataFrame with
    one row per trial, best first.
    """
    resampled = resample(X_train, y_train, strategy, ratio)
    sampler = ParameterSampler(SEARCH_SPACE, n_iter=n_trials, random_state=seed)

    rows, best_model, best_score = [], None, -np.inf
    start = time.time()
    for trial, params in enumerate(sampler, start=1):
        if time.time() - start > time_budget_s:
            print(f"Time budget reached after {trial - 1} trials.")
            break
        params = {k: (int(v) if k == "max_depth" else float(v)) for k, v in params.items()}
        t0 = time.time()
        model = train_xgb(X_train, y_train, X_val, y_val, strategy, params, ratio, resampled)
        score = average_precision_score(y_val, model.predict_proba(X_val)[:, 1])
        rows.append({"trial": trial, **params, "best_iteration": int(model.best_iteration),
                     "val_pr_auc": score, "seconds": round(time.time() - t0, 1)})
        print(f"trial {trial:>2}: val PR-AUC {score:.4f} "
              f"({rows[-1]['seconds']:.0f}s, {model.best_iteration + 1} trees)")
        if score > best_score:
            best_model, best_score = model, score

    results = (pd.DataFrame(rows)
               .sort_values("val_pr_auc", ascending=False, kind="stable")  # ties keep trial order, matching best_model
               .reset_index(drop=True))
    return best_model, results


def save_model(model, name: str) -> Path:
    """Save a fitted model to models/<name>.joblib (the folder is gitignored)."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def load_model(name: str):
    """Load a model saved with :func:`save_model`."""
    return joblib.load(MODEL_DIR / f"{name}.joblib")
