from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW_CSV = ROOT / "data" / "online_shoppers_intention.csv"
HOLD_OUT_CSV = ROOT / "test_data.csv"
SCORES_PATH = HERE / "scores.json"

LABEL = "Revenue"
SEED = 17
TEST_FRAC = 0.25

# Session-behaviour counters / rates stay numeric.
NUMERIC = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]

# Coded integers are categories, not magnitudes.
CATEGORICAL = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

CLASSIFIERS = {
    "logreg": LogisticRegression(
        C=1.4, solver="liblinear", max_iter=1500, random_state=SEED
    ),
    "dtree": DecisionTreeClassifier(
        max_depth=8, min_samples_split=25, min_samples_leaf=8, random_state=SEED
    ),
    "knn": KNeighborsClassifier(n_neighbors=9, weights="distance"),
    "nbayes": GaussianNB(var_smoothing=1e-8),
    "rforest": RandomForestClassifier(
        n_estimators=180,
        max_depth=16,
        min_samples_leaf=4,
        random_state=SEED,
        n_jobs=-1,
    ),
}


def coerce_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Make dtypes stable so a CSV upload matches the fitted preprocessor."""
    cleaned = table.copy()
    for col in CATEGORICAL:
        if col not in cleaned.columns:
            continue
        if col == "Weekend":
            cleaned[col] = (
                cleaned[col]
                .astype(str)
                .str.lower()
                .isin(["true", "1", "yes"])
                .map({True: "weekend", False: "weekday"})
            )
        else:
            cleaned[col] = cleaned[col].astype(str)
    return cleaned


def encode_label(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)
    return series.astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)


def make_preprocessor() -> ColumnTransformer:
    numeric_branch = Pipeline(
        steps=[
            ("fill", SimpleImputer(strategy="median")),
            ("zscore", StandardScaler()),
        ]
    )
    category_branch = Pipeline(
        steps=[
            ("fill", SimpleImputer(strategy="most_frequent")),
            ("dummies", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_branch, NUMERIC),
            ("category", category_branch, CATEGORICAL),
        ]
    )


def six_scores(fitted: Pipeline, features: pd.DataFrame, truth: pd.Series) -> dict:
    pred = fitted.predict(features)
    proba = fitted.predict_proba(features)[:, 1]
    return {
        "accuracy": round(accuracy_score(truth, pred), 4),
        "auc": round(roc_auc_score(truth, proba), 4),
        "precision": round(precision_score(truth, pred, zero_division=0), 4),
        "recall": round(recall_score(truth, pred, zero_division=0), 4),
        "f1": round(f1_score(truth, pred, zero_division=0), 4),
        "mcc": round(matthews_corrcoef(truth, pred), 4),
    }


def main() -> None:
    raw = pd.read_csv(RAW_CSV)
    print(f"loaded {RAW_CSV.name}: {raw.shape[0]} rows x {raw.shape[1]} cols")

    features = coerce_columns(raw.drop(columns=[LABEL]))
    truth = encode_label(raw[LABEL])

    x_fit, x_hold, y_fit, y_hold = train_test_split(
        features,
        truth,
        test_size=TEST_FRAC,
        stratify=truth,
        random_state=SEED,
    )

    hold_dump = x_hold.copy()
    hold_dump[LABEL] = raw.loc[x_hold.index, LABEL]
    hold_dump.to_csv(HOLD_OUT_CSV, index=False)
    print(
        f"wrote hold-out split -> {HOLD_OUT_CSV.name} ({len(hold_dump)} rows)")

    board = {}
    for key, clf in CLASSIFIERS.items():
        pipe = Pipeline(steps=[("prep", make_preprocessor()), ("clf", clf)])
        pipe.fit(x_fit, y_fit)
        board[key] = six_scores(pipe, x_hold, y_hold)
        out = HERE / f"{key}.joblib"
        joblib.dump(pipe, out, compress=3)
        print(f"{key:8s} {board[key]}  -> {out.name}")

    SCORES_PATH.write_text(json.dumps(board, indent=2))
    print(f"wrote {SCORES_PATH.name}")


if __name__ == "__main__":
    main()
