from __future__ import annotations

from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

REPO = Path(__file__).resolve().parent
MODEL_STORE = REPO / "model"
HOLD_OUT = REPO / "test_data.csv"
LABEL_COL = "Revenue"

COUNT_COLS = [
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
CODED_COLS = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]
INPUT_COLS = COUNT_COLS + CODED_COLS

ESTIMATORS = {
    "Logistic Regression": "logreg",
    "Decision Tree": "dtree",
    "k-Nearest Neighbours": "knn",
    "Gaussian Naive Bayes": "nbayes",
    "Random Forest Ensemble": "rforest",
}


def normalise_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in CODED_COLS:
        if col not in out.columns:
            continue
        if col == "Weekend":
            out[col] = (
                out[col]
                .astype(str)
                .str.lower()
                .isin(["true", "1", "yes", "weekend"])
                .map({True: "weekend", False: "weekday"})
            )
        else:
            out[col] = out[col].astype(str)
    return out


def as_binary(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)
    return series.astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)


@st.cache_resource
def estimator(slug: str):
    return joblib.load(MODEL_STORE / f"{slug}.joblib")


@st.cache_data
def bundled_holdout() -> pd.DataFrame:
    return pd.read_csv(HOLD_OUT)


def ingest_csv(upload) -> pd.DataFrame:
    table = pd.read_csv(upload)
    if table.shape[1] == 1:
        upload.seek(0)
        table = pd.read_csv(upload, sep=";")
    return table


def scoreboard(truth, pred, proba) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(truth, pred),
        "AUC": roc_auc_score(truth, proba),
        "Precision": precision_score(truth, pred, zero_division=0),
        "Recall": recall_score(truth, pred, zero_division=0),
        "F1": f1_score(truth, pred, zero_division=0),
        "MCC": matthews_corrcoef(truth, pred),
    }


def confusion_heatmap(truth, pred) -> alt.Chart:
    matrix = confusion_matrix(truth, pred)
    names = ["no purchase", "purchase"]
    cells = [
        {"actual": names[i], "predicted": names[j], "count": int(matrix[i, j])}
        for i in range(2)
        for j in range(2)
    ]
    src = pd.DataFrame(cells)
    tiles = (
        alt.Chart(src)
        .mark_rect(cornerRadius=4)
        .encode(
            x=alt.X("predicted:N", title="Predicted", sort=names),
            y=alt.Y("actual:N", title="Actual", sort=list(reversed(names))),
            color=alt.Color(
                "count:Q",
                scale=alt.Scale(scheme="oranges"),
                legend=None,
            ),
            tooltip=["actual", "predicted", "count"],
        )
        .properties(width=340, height=300, title="Confusion matrix")
    )
    labels = (
        alt.Chart(src)
        .mark_text(fontSize=18, fontWeight=700)
        .encode(
            x=alt.X("predicted:N", sort=names),
            y=alt.Y("actual:N", sort=list(reversed(names))),
            text="count:Q",
            color=alt.value("#1c1917"),
        )
    )
    return tiles + labels


def roc_line(truth, proba, auc_value: float) -> alt.Chart:
    fpr, tpr, _ = roc_curve(truth, proba)
    curve = pd.DataFrame(
        {"false_positive_rate": fpr, "true_positive_rate": tpr})
    chance = pd.DataFrame(
        {"false_positive_rate": [0, 1], "true_positive_rate": [0, 1]}
    )
    model_line = (
        alt.Chart(curve)
        .mark_line(strokeWidth=2.5, color="#f59e0b")
        .encode(
            x=alt.X("false_positive_rate:Q", title="False positive rate",
                    scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("true_positive_rate:Q", title="True positive rate",
                    scale=alt.Scale(domain=[0, 1])),
        )
    )
    baseline = (
        alt.Chart(chance)
        .mark_line(strokeDash=[5, 5], color="#9ca3af")
        .encode(x="false_positive_rate:Q", y="true_positive_rate:Q")
    )
    return (baseline + model_line).properties(
        width=340, height=300, title=f"ROC  ·  AUC {auc_value:.3f}"
    )


def conversion_bars(truth, pred) -> alt.Chart:
    """Grouped bars, faceted by outcome, so the rare purchase class is readable."""
    src = pd.DataFrame(
        {
            "source": ["Actual labels", "Actual labels", "Model output", "Model output"],
            "outcome": ["Purchase", "No purchase", "Purchase", "No purchase"],
            "sessions": [
                int(truth.sum()),
                int(len(truth) - truth.sum()),
                int(pred.sum()),
                int(len(pred) - pred.sum()),
            ],
        }
    )
    dark = getattr(getattr(st.context, "theme", None),
                   "type", "light") == "dark"
    label_color = "#f3f4f6" if dark else "#1c1917"
    bars = (
        alt.Chart(src)
        .mark_bar(size=46, cornerRadiusEnd=4)
        .encode(
            x=alt.X("source:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("sessions:Q", title="Sessions"),
            color=alt.Color(
                "source:N",
                scale=alt.Scale(
                    domain=["Actual labels", "Model output"],
                    range=["#6b7280", "#f59e0b"],
                ),
                legend=None,
            ),
            tooltip=["source", "outcome", "sessions"],
        )
    )
    numbers = (
        alt.Chart(src)
        .mark_text(dy=-8, fontSize=13, fontWeight=600, color=label_color)
        .encode(
            x="source:N",
            y="sessions:Q",
            text=alt.Text("sessions:Q", format=","),
        )
    )
    return (
        (bars + numbers)
        .properties(width=200, height=240)
        .facet(column=alt.Column("outcome:N", title=None))
        .resolve_scale(y="independent")
    )


def main() -> None:
    st.set_page_config(
        page_title="Conversion Scoring Desk",
        page_icon=":material/shopping_bag:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    title_col, badge_col = st.columns([3.2, 1.6], vertical_alignment="center")
    with title_col:
        st.header("Conversion Scoring Desk")
        st.caption(
            "Anchit Khare · 2025AC05473 · BITS WILP M.Tech (AIML/DSE) · Assignment 2 · "
            "UCI Online Shoppers Purchasing Intention"
        )
    with badge_col:
        st.badge("17 features", icon=":material/view_column:", color="orange")
        st.badge("binary target: Revenue",
                 icon=":material/flag:", color="blue")

    pick_col, file_col = st.columns([1.1, 1.6], vertical_alignment="bottom")
    with pick_col:
        model_name = st.selectbox(
            "Choose a classifier",
            options=list(ESTIMATORS),
            index=4,
            help="All five assignment models, trained on the same 75% split.",
        )
    with file_col:
        upload = st.file_uploader(
            "Upload test data (CSV)",
            type=["csv"],
            help="Must include the 17 session columns. "
            f"Add `{LABEL_COL}` to unlock metrics, the confusion matrix and the report.",
        )

    if upload is not None:
        sessions = ingest_csv(upload)
        origin = upload.name
        origin_kind = "upload"
    else:
        sessions = bundled_holdout()
        origin = "test_data.csv (bundled 25% hold-out)"
        origin_kind = "bundled"

    missing = [c for c in INPUT_COLS if c not in sessions.columns]
    if missing:
        st.error(f"This CSV is missing required columns: {missing}")
        st.stop()

    fitted = estimator(ESTIMATORS[model_name])
    features = normalise_inputs(sessions[INPUT_COLS])
    pred = fitted.predict(features)
    proba = fitted.predict_proba(features)[:, 1]
    labelled = LABEL_COL in sessions.columns

    stat1, stat2, stat3, stat4 = st.columns(4)
    stat1.metric("Sessions scored", f"{len(sessions):,}")
    stat2.metric("Predicted purchases", f"{int(pred.sum()):,}")
    if labelled:
        truth = as_binary(sessions[LABEL_COL])
        stat3.metric("Actual purchases", f"{int(truth.sum()):,}")
        stat4.metric("Actual conversion", f"{truth.mean():.1%}")
    else:
        stat3.metric("Actual purchases", "—")
        stat4.metric("Labels in file", "no")
    st.caption(
        f"Source: **{origin}** ({origin_kind}) · classifier: **{model_name}**"
    )

    with st.expander("Sample of the loaded rows", expanded=False):
        st.dataframe(sessions.head(12), width="stretch", hide_index=True)

    if labelled:
        truth = as_binary(sessions[LABEL_COL])
        marks = scoreboard(truth, pred, proba)

        st.subheader("Evaluation metrics")
        with st.container(border=True):
            cells = st.columns(6)
            for cell, name in zip(cells, marks):
                cell.metric(name, f"{marks[name]:.4f}")

        chart_l, chart_r = st.columns(2)
        with chart_l:
            st.altair_chart(confusion_heatmap(truth, pred), width="stretch")
        with chart_r:
            st.altair_chart(
                roc_line(truth, proba, marks["AUC"]), width="stretch")

        st.markdown("**Classification report**")
        report = classification_report(
            truth,
            pred,
            target_names=["no purchase", "purchase"],
            output_dict=True,
            zero_division=0,
        )
        st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")

        st.markdown("**Actual vs model conversion mix**")
        st.caption(
            "Purchase and no-purchase use separate y-scales so the smaller purchase "
            "class is not flattened by the majority class."
        )
        st.altair_chart(conversion_bars(truth, pred), width="content")
    else:
        st.warning(
            f"No `{LABEL_COL}` column in this file — metrics, confusion matrix "
            "and classification report need ground-truth labels. Predictions below."
        )

    st.subheader("Scored sessions")
    only_hits = st.toggle("Show predicted purchases only", value=False)
    scored = sessions.copy()
    scored["predicted_revenue"] = pd.Series(
        pred).map({0: False, 1: True}).values
    scored["p_purchase"] = proba.round(4)
    shown = scored[scored["predicted_revenue"]] if only_hits else scored
    st.dataframe(shown.head(60), width="stretch", hide_index=True)
    st.download_button(
        label="Download scored CSV",
        data=scored.to_csv(index=False).encode(),
        file_name="scored_sessions.csv",
        mime="text/csv",
        icon=":material/download:",
    )


if __name__ == "__main__":
    main()
