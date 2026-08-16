# Online Shoppers — Purchase Intention (ML Assignment 2)

M.Tech (AIML/DSE) · Machine Learning · BITS Pilani WILP

**Name:** Anchit Khare &nbsp;·&nbsp; **BITS ID:** 2025AC05473 &nbsp;|&nbsp; **Email:** 2025ac05473@wilp.bits-pilani.ac.in

## a. Problem Statement

An e-commerce site logs every browsing session: which kinds of pages were opened,
how long the visitor stayed, bounce/exit rates, traffic source, and whether the
visit fell near a campaign day. Most sessions still end without a sale, so ranking
sessions by conversion likelihood is useful for live offers and remarketing.

This project trains and compares five classification models that predict
**whether a session will generate revenue (`Revenue` = True / False)** from those
session logs. All six assignment metrics are reported on a held-out split, and the
fitted pipelines are exposed through a Streamlit app on Streamlit Community Cloud.

## b. Dataset Description

- **Source:** [UCI Machine Learning Repository — Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
- **Task type:** Binary classification
- **Instances:** 12,330 sessions (assignment minimum: 500 ✔)
- **Features:** 17 input features (assignment minimum: 12 ✔) + 1 target
- **Target:** `Revenue` — did the session end in a purchase? (`True` / `False`)
- **Class balance:** imbalanced — only ≈ 15.5% of sessions converted

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | Administrative | numeric | Count of administrative pages opened |
| 2 | Administrative_Duration | numeric | Seconds spent on administrative pages |
| 3 | Informational | numeric | Count of informational pages opened |
| 4 | Informational_Duration | numeric | Seconds spent on informational pages |
| 5 | ProductRelated | numeric | Count of product pages opened |
| 6 | ProductRelated_Duration | numeric | Seconds spent on product pages |
| 7 | BounceRates | numeric | Average bounce rate of the visited pages |
| 8 | ExitRates | numeric | Average exit rate of the visited pages |
| 9 | PageValues | numeric | Average page value of the visited pages |
| 10 | SpecialDay | numeric | Closeness of the visit to a special day (0–1) |
| 11 | Month | categorical | Month of the visit |
| 12 | OperatingSystems | categorical | OS identifier (coded integer, treated as a category) |
| 13 | Browser | categorical | Browser identifier |
| 14 | Region | categorical | Visitor region |
| 15 | TrafficType | categorical | Traffic-source type |
| 16 | VisitorType | categorical | New / returning / other |
| 17 | Weekend | categorical | Weekend vs weekday visit |

**Preprocessing:** numeric columns are median-imputed and standardised; categorical
columns (including the coded integers and weekend flag) are mode-imputed and
one-hot encoded. Both steps live inside a scikit-learn `Pipeline`, so the saved
`.joblib` files accept raw CSV rows. Split: 75% train / 25% test, stratified on
`Revenue`, `random_state=17`. The 3,083-row hold-out file is `test_data.csv`.

## c. GitHub Repository Link

👉 **https://github.com/anchit-khare/ml-assignment-streamlit**

Repository contents:

```
├── app.py                              # Streamlit frontend
├── requirements.txt                    # Python dependencies
├── README.md                           # this file
├── test_data.csv                       # held-out 25% split (experiments + app default)
├── data/
│   └── online_shoppers_intention.csv   # raw UCI dataset
└── model/
    ├── fit.py                          # trains all 5 classifiers
    ├── scores.json                     # evaluation metrics
    └── *.joblib                        # 5 fitted pipelines
```

**Live Streamlit App:** 👉 **https://anchit-khare-ml-assignment-2.streamlit.app**

## d. Models Used — Comparison Table

All five models were fit on the same 75% training split and scored on the same
3,083-row hold-out split of `online_shoppers_intention.csv`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8806 | 0.8842 | 0.7088 | 0.3878 | 0.5014 | 0.4660 |
| Decision Tree | 0.8910 | 0.8845 | 0.6911 | 0.5346 | 0.6028 | 0.5469 |
| kNN | 0.8725 | 0.8052 | 0.6654 | 0.3543 | 0.4624 | 0.4231 |
| Naive Bayes | 0.2887 | 0.7853 | 0.1765 | 0.9811 | 0.2991 | 0.1492 |
| Random Forest (Ensemble) | 0.8978 | 0.9231 | 0.7596 | 0.4969 | 0.6008 | 0.5613 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Solid linear baseline: AUC 0.8842 is almost identical to the single tree, which suggests a large share of the signal (especially `PageValues` and bounce/exit rates) is linearly usable after scaling. Precision is decent (0.71) but recall is only 0.39 — the default 0.5 threshold is conservative on a 15.5% positive class, so many converting sessions are missed. |
| Decision Tree | Depth capped at 8. Best F1 (0.6028) and the highest recall among the precise models (0.53). Accuracy (0.8910) already beats logistic regression, but AUC stays ~0.88 because a single tree's probability estimates are coarse (leaf frequencies). It is the most “balanced” non-ensemble model on this data. |
| kNN | Weakest of the distance/linear/tree group: lowest AUC (0.8052), recall (0.35) and F1 (0.4624). After one-hot encoding, the feature space is high-dimensional and sparse; 9-NN with distance weights still lets the majority “no purchase” neighbours dominate, so converters are under-detected. |
| Naive Bayes | Collapses on accuracy (0.2887) while posting almost-perfect recall (0.98). GaussianNB is a poor match for one-hot 0/1 columns and for tightly related pairs such as `BounceRates`/`ExitRates` and page-count vs duration. It scores nearly every session as a purchase, which tanks precision (0.18) and MCC (0.15). AUC 0.7853 shows *ranking* is not hopeless — the class prior / threshold is. |
| Random Forest (Ensemble) | Best overall: highest accuracy (0.8978), AUC (0.9231), precision (0.7596) and MCC (0.5613). Bagging 180 depth-limited trees recovers the interactions a linear model cannot (e.g. high `PageValues` in November × returning visitor) and calibrates probabilities better than one tree (AUC jump 0.88 → 0.92). F1 (0.6008) is a hair behind the single tree because the forest is slightly more conservative on recall. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** — it leads on 4 of 6 metrics, and MCC (the fairest single number on a 15.5% positive class) is clearly highest at 0.5613. If the business goal were “catch almost every buyer” regardless of extra offers sent, the Decision Tree (recall 0.53, best F1) is the practical runner-up; Naive Bayes’s 0.98 recall is not usable because of the flood of false positives. |

**General note:** a dummy classifier that always predicts “no purchase” would already
score ≈ 0.845 accuracy. That is why Accuracy looks uniformly high (except Naive
Bayes) while F1 / MCC spread the models apart. AUC is the right lens for ranking
sessions for a marketing campaign.

## How to Run Locally

```bash
pip install -r requirements.txt
python model/fit.py        # optional — .joblib files are already in model/
streamlit run app.py
```

## Streamlit App Features

1. **CSV upload** of test data in the main toolbar (defaults to bundled `test_data.csv`)
2. **Model selection dropdown** — all 5 fitted classifiers
3. **Evaluation metrics** — Accuracy, AUC, Precision, Recall, F1, MCC on a scoreboard
4. **Confusion matrix** (Altair heatmap) and **classification report**, plus an ROC curve
5. Conversion-mix chart, scored-session table, and a downloadable predictions CSV
