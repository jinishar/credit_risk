# AI-Powered Credit Risk Intelligence Platform

A lightweight, end-to-end AI platform on the Kaggle **Home Credit Default Risk**
dataset: EDA → an imbalance-aware ML default-risk model → SHAP explainability
→ business rule derivation → a Gemini-powered talk-to-data chatbot → one
multi-tab Streamlit UI → Docker Compose deployment.

> **Data.** All numbers below are from the **real Kaggle Home Credit
> `application_train.csv`** — 307,511 applicants, 122 columns, 8.07% default
> rate (≈11.4:1 imbalance). The CSVs are not committed to git (per the
> assignment's code structure) and are mounted into the container at runtime.
> If `data/application_train.csv` is absent, `src/data/loader.py`
> automatically falls back to a **schema-accurate synthetic generator**
> (`src/data/synthetic.py`) so `docker-compose up` still produces a fully
> working demo with zero setup — see
> [Getting the real dataset](#getting-the-real-dataset-and-the-synthetic-fallback).

## Architecture

```
                         ┌─────────────────────────┐
                         │   data/application_*.csv │  (real Kaggle CSV if present,
                         │   (real or synthetic)    │   else synthetic.py generates it)
                         └────────────┬─────────────┘
                                      │
                     ┌────────────────┴─────────────────┐
                     ▼                                   ▼
          src/data/loader.py                    src/data/loader.py
        (pandas, for ML pipeline)              (DuckDB, for chatbot)
                     │                                   │
                     ▼                                   ▼
        src/data/preprocessor.py            sql/schema.sql (grounding context)
      (encode/feature-engineer,                          │
       keep NaN + missing-flags)                         ▼
                     │                     src/talk_to_data/
                     ▼                       prompt_templates.py  (schema + few-shot)
        src/ml/train.py                      nl_to_sql.py         (Gemini + validator)
      (LogReg baseline + LightGBM             query_runner.py      (DuckDB exec + memory)
       champion, isotonic                               │
       calibration) → models/*.joblib                    │
                     │                                   │
        ┌────────────┼────────────┐                      │
        ▼            ▼            ▼                      │
 src/ml/predict  src/ml/explain  src/rules/               │
  (score + band)   (SHAP)      rule_engine.py              │
        │            │        (surrogate tree +            │
        │            │         threshold bins)              │
        └──────┬─────┴───────────────┬────────────────────┘
               ▼                     ▼
         app/streamlit_app.py  (EDA · Prediction · Explainability · Rules · Chatbot)
               │
               ▼
         Docker Compose (single `app` service, DuckDB + volumes for data/models)
```

**Why these choices:**
- **DuckDB, not Postgres** — the chatbot needs SQL, but there's no reason to run
  a second container for it; DuckDB is embedded, fast for analytical
  queries, and reads the same CSV the ML pipeline uses, so both layers stay
  perfectly in sync with zero duplication.
- **LightGBM + Logistic Regression baseline, not deep learning** — the data
  is medium-sized tabular (307K rows) with heavy missingness and mixed types;
  LightGBM handles both natively (including `NaN` splits), trains in seconds
  (fits "lightweight platform"), and is exactly explainable via SHAP's fast
  `TreeExplainer`. Logistic Regression is trained alongside as an
  interpretable sanity-check baseline; the model with the higher validation
  ROC-AUC is promoted automatically.
- **Folder-tree deviations** — `src/ml/explain.py`, `src/rules/`, and `app/`
  are added because the assignment's published tree
  (`Docs/NeoStats_AI_Use_Case.pdf`) has no slot for the SHAP module, the
  rule-derivation module, or the UI code, all three of which are separately
  required deliverables. Every other path matches the tree exactly.

## Setup & Run

### Option A — Docker (recommended, one command)

```bash
cp .env.example .env        # optionally add GEMINI_API_KEY for the chatbot
docker-compose up
```

Open **http://localhost:8501**. To run on the real Kaggle data, drop
`application_train.csv` / `application_test.csv` into `data/` **before**
`docker-compose up` (see [below](#getting-the-real-dataset-and-the-synthetic-fallback));
otherwise the container generates a synthetic dataset so the demo still
works with no download. On first start it trains the model and derives
business rules (~1-2 min on the real data); subsequent restarts reuse the
artifacts in the mounted `models/` volume and start instantly.

### Option B — Local Python

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env        # optionally add GEMINI_API_KEY
python -m src.ml.train
python -m src.rules.rule_engine
streamlit run app/streamlit_app.py
```

### Getting the real dataset (and the synthetic fallback)

The real CSVs aren't in the repo. To run the platform exactly as documented:

1. Download `application_train.csv` / `application_test.csv` from the
   [Kaggle competition](https://www.kaggle.com/competitions/home-credit-default-risk/data)
   (accept the rules, then *Download All*).
2. Put both files in `data/`.
3. `rm -f models/* data/credit_risk.duckdb` so the pipeline retrains on them.

If you skip this, `src/data/loader.py` calls `src/data/synthetic.py` to
generate a schema-accurate stand-in (122 columns, same imbalance and the
`DAYS_EMPLOYED == 365243` quirk) so the app still runs end to end — the
model/EDA numbers will differ from those below.

## Model Selection & Class Imbalance Strategy

Two models are trained and compared on a stratified 80/20 split (246,008
train / 61,503 validation, 236 engineered features):

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic Regression — median-impute + standardize + `class_weight='balanced'` — baseline | 0.750 | 0.235 |
| **LightGBM (320 trees) + isotonic calibration — champion** | **0.763** | 0.246 |

LightGBM is promoted automatically (higher validation ROC-AUC). Chosen over a
deep model because the data is tabular with heavy missingness and mixed
numeric/categorical types — LightGBM handles both natively, trains in
seconds, and pairs with a fast, exact SHAP `TreeExplainer`. ~0.76 ROC-AUC is
in the expected range for a model built on `application_train` alone (the
bureau / previous-application tables, not used here, are what take Kaggle
leaderboard solutions to ~0.79+).

**Missing values.** The preprocessor deliberately does **not** impute numeric
features — LightGBM splits on `NaN` natively, and the strongest predictors
(`EXT_SOURCE_1/2/3`) are 40–55% missing, so imputing them flattens the
signal. Instead a `<col>_MISSING` indicator is added for every numeric column
that was materially missing at fit time. The Logistic Regression baseline
can't take `NaN`, so it carries its own `SimpleImputer` + `StandardScaler`
pipeline (without scaling, lbfgs doesn't converge here).

**Imbalance handling** (8.07% positive class, ≈11.4:1):
- Logistic Regression: `class_weight='balanced'`.
- LightGBM: `scale_pos_weight` and `is_unbalance` were both tested and
  **rejected** — the reweighted objective spikes validation AUC on the first
  1–3 trees and then plateaus, so early stopping cuts training off almost
  immediately (AUC ~0.73 vs ~0.76 without). Imbalance is instead handled
  downstream: threshold-independent headline metrics (ROC-AUC / PR-AUC), and
  isotonic calibration (below) anchoring probabilities to the true 8% base
  rate, with risk bands read off the calibrated probability rather than a
  fixed 0.5 cutoff.
- No SMOTE — see [Known Limitations](#known-limitations--improvements).

**Probability calibration.** LightGBM's raw scores rank well but sit far below
the true scale under this imbalance (the "High" band >0.5 would almost never
fire). `CalibratedClassifierCV(method="isotonic", cv="prefit")` on the
held-out validation fold remaps them: mean predicted probability **0.081** vs
actual base rate **0.079**, and the bands become meaningful — actual default
rate is **5.5%** in the Low band, **34.6%** in Medium, **68.5%** in High
(distribution: 92% / 7% / 0.4% of applicants).

**Risk bands** (`src/utils/helpers.py::risk_band`, thresholds in `.env`):
Low `< 0.2`, Medium `0.2 – 0.5`, High `> 0.5`.

## Evaluation Metrics & Results

From `models/metrics.json` (regenerate with `python -m src.ml.train`):

```
Champion: LightGBM + isotonic calibration   (val: 61,503 rows, 8.07% positive)
  ROC-AUC:    0.763
  PR-AUC:     0.246
  F1 @ 0.5:   0.059   (precision 0.595, recall 0.031)
  Confusion matrix @ 0.5: TN=56433  FP=105  FN=4811  TP=154
  Logistic Regression baseline: ROC-AUC 0.750 / PR-AUC 0.235
```

ROC-AUC/PR-AUC are the headline metrics rather than accuracy or a fixed-0.5
F1: with a 92/8 class split, a model that always predicts "no default"
scores ~92% accuracy while being useless, and the F1 @ 0.5 above is
near-zero precisely because 0.5 is the wrong threshold for an 8%-prevalence
problem. The *actual* decision threshold a bank uses should be set from
business risk appetite (a precision/recall trade-off) — which is why the
risk bands operate on the calibrated probability instead of a single 0.5
cutoff, and why calibration quality (see above) matters more here than the
0.5 confusion matrix.

## Explainable AI

SHAP `TreeExplainer` (`src/ml/explain.py`) on the underlying LightGBM
booster (unwrapped from the calibration wrapper for exact tree attributions).
Chosen over LIME because it's exact (no local sampling) and fast enough for
interactive per-applicant explanations, and gives both a global
feature-importance view and a signed, additive per-prediction breakdown from
the same explainer.

Top global drivers found (mean |SHAP|): `EXT_SOURCE_MEAN` (dominant, ~3x the
next), `EXT_SOURCE_3`, `GOODS_PRICE_CREDIT_RATIO`, `AMT_ANNUITY`,
`AMT_GOODS_PRICE`, `EXT_SOURCE_1`, `DAYS_EMPLOYED`, education level — all
business-sensible (external credit-bureau scores and loan affordability). The
Explainability tab shows the top contributing features for whichever
applicant was just scored, each labeled "increases risk" / "decreases risk"
rather than exposing raw SHAP log-odds numbers, which would be meaningless to
a non-technical user.

## Business Rule Derivation

`src/rules/rule_engine.py` produces two complementary, audit-friendly rule
types (spec's "bridge ML insights and credit policy with rules"):

1. **Surrogate decision tree** (depth ≤3) trained to mimic the champion
   model's predictions — every root-to-leaf path is a readable IF/THEN rule
   a credit analyst can review, e.g.:
   ```
   IF EXT_SOURCE_MEAN <= 0.25 AND EXT_SOURCE_3 <= 0.226
   → predicted default rate 34.1% (384 applicants)
   ```
   ```
   IF EXT_SOURCE_MEAN > 0.643
   → predicted default rate 2.4% (3,991 applicants)
   ```
2. **Threshold bins** on the model's dominant driver (`EXT_SOURCE_MEAN`),
   reporting the *actual observed* default rate per quantile (computed against
   the real `TARGET` labels, not model predictions) — grounded directly in
   data:
   ```
   EXT_SOURCE_MEAN in (0.00, 0.38]  → observed default rate 19.9%
   EXT_SOURCE_MEAN in (0.64, 0.86]  → observed default rate  2.6%
   ```

Both rule sets are shown in the UI's Business Rules tab, alongside which
rules the currently-scored applicant triggers.

## Talk-to-Data: Prompt Engineering & Token Optimization

`src/talk_to_data/` (Gemini via `google-genai`):

- **Two-tier answering** (`nl_to_sql.py`): the 5 required guaranteed query
  patterns are matched by keyword *before* ever calling the LLM, so they're
  100% reliable and hallucination-free. Everything else falls through to
  Gemini. This is the main hallucination-control mechanism — deterministic
  answers where correctness matters most, LLM only for genuine long-tail
  questions.
- **SQL validation** (`nl_to_sql.py::validate_sql`): every LLM-generated
  query is parsed with `sqlparse` and rejected unless it is a single
  `SELECT` statement referencing only the whitelisted `applications` table,
  with no DDL/DML/administrative keywords. Tested against
  `DROP TABLE applications`, stacked `SELECT ...; DELETE ...`, and
  cross-table access — all correctly rejected (see `nl_to_sql.py`'s
  `__main__` block). `query_runner.py` also caps result size, appending
  `LIMIT 200` if the query didn't already include one.
- **Token optimization**: the system prompt sends a curated ~25-column
  schema summary (`prompt_templates.SCHEMA_CONTEXT`), not the full
  121-column dictionary; conversation memory sent on follow-ups is a list of
  prior `(question, sql)` pairs capped at the last 4 turns, never raw result
  tables; and result narration is built from a truncated preview (≤15 rows)
  rather than the full result set.
- **Conversation memory**: `ChatSession.history` (in `query_runner.py`)
  lets follow-ups like "and for women only?" resolve against the previous
  question's SQL.
- **Graceful degradation without a key**: with no `GEMINI_API_KEY` set, the
  5 canonical questions still run end-to-end (SQL executes, and narration
  falls back to a plain templated summary of the result); only genuinely
  free-form questions require the key.

### Sample chatbot output (real data, no Gemini key needed for these)

| Question | SQL generated | Result |
|---|---|---|
| What is the overall default rate? | `SELECT AVG(TARGET)*100 AS default_rate_pct, COUNT(*) AS n_applicants FROM applications` | 8.07%, 307,511 applicants |
| How many applicants own both a car and a house? | `SELECT COUNT(*) FROM applications WHERE FLAG_OWN_CAR='Y' AND FLAG_OWN_REALTY='Y'` | 72,360 |
| Show default rate by education level | `SELECT NAME_EDUCATION_TYPE, AVG(TARGET)*100 ... GROUP BY ...` | Lower secondary 10.9% → Academic degree 1.8% |

With a real `GEMINI_API_KEY`, free-form questions (e.g. "what's the median
annuity for revolving loans?" → `SELECT MEDIAN(AMT_ANNUITY) ... WHERE
NAME_CONTRACT_TYPE = 'Revolving loans'`) are additionally supported through
the validated Gemini path (~15–20 s per call, Gemini-side latency).

## EDA — Key Findings

Full analysis in `notebooks/eda.py` / `notebooks/eda.ipynb`, figures in
`notebooks/figures/`. Headline findings on the real dataset (307,511 rows,
122 columns, 8.07% default rate, 11.4:1 imbalance):

1. **Default rate by education**: 1.8% (Academic degree) → 5.4% (Higher
   education) → 10.9% (Lower secondary) — a clean monotonic gradient.
2. **External credit scores** (`EXT_SOURCE_1/2/3`) are the strongest
   continuous predictors, correlating −0.16 / −0.16 / −0.18 with `TARGET`;
   their mean (`EXT_SOURCE_MEAN`) is the single dominant feature for both the
   model and the derived rules.
3. **Age**: default rate falls steadily from 12.3% (18–25) to 3.7% (65+).
4. **Credit-to-income ratio is *not* predictive on its own** — default rate
   is a flat ~7–9% across all five quintiles. (It matters only in
   interaction with `EXT_SOURCE`, which the tree model picks up.) A useful
   negative result: the "obvious" affordability ratio isn't a standalone
   lever here.
5. **Family status**: Widowed 5.8% and Married 7.6% default less than
   Single 9.8% / Civil-marriage 9.9%.
6. **Data quality**: 41 of 122 columns are >50% missing, dominated by the
   building/apartment block (`COMMONAREA_*`, `NONLIVINGAPARTMENTS_*` etc.,
   ~68–70% each). `DAYS_EMPLOYED` holds the well-known `365243` placeholder
   ("not currently employed", mostly pensioners) in 18.0% of rows — those
   applicants actually default *less* (5.4% vs 8.7%); handled in
   `preprocessor.py` as an explicit anomaly flag, not a real day count.

## Known Limitations & Improvements

- **No bureau/previous-application tables.** Only `application_train/test`
  is used — a deliberate scope decision for a "lightweight platform". This is
  the main reason ROC-AUC sits at ~0.76 rather than ~0.79+: joining
  `bureau.csv`, `previous_application.csv`, `installments_payments.csv` etc.
  and aggregating repayment history per applicant is where the remaining
  signal is, at the cost of significantly more ETL.
- **Calibration fit on the same fold it's evaluated on** is mildly
  optimistic (`CalibratedClassifierCV(cv="prefit")` on `X_val`/`y_val`, then
  evaluated on that same fold). With more data, calibrate on a third
  held-out split distinct from both training and evaluation.
- **No SMOTE/oversampling.** Class weighting (baseline) + calibration
  (champion) were used instead — they keep the pipeline lightweight and the
  probabilities directly calibratable. SMOTE / borderline-SMOTE on the
  training fold only (never validation/test) is worth A/B-testing.
- **No hyperparameter search.** The LightGBM params are hand-set to
  sensible Home-Credit defaults; a small Optuna/`RandomizedSearchCV` pass
  would likely add ~0.005–0.01 ROC-AUC.
- **Chatbot free-form questions require a Gemini key** and take ~15–20 s
  (Gemini-side latency). The 5 canonical patterns are instant and need no
  key; add `GEMINI_API_KEY` to `.env` to unlock open-ended questions.
- **Synthetic fallback.** If the real CSVs aren't in `data/`, the platform
  runs on generated data (`src/data/synthetic.py`) — the app works but the
  model/EDA numbers won't match this README.
- **Surrogate-tree rules approximate, not exactly reproduce, the LightGBM
  model.** They're intentionally shallow (depth ≤3) for human readability;
  treat them as policy guidance, not a drop-in replacement for the model's
  own score.

## Project Structure

```
credit_risk_platform/
├── data/                  # gitignored - real Kaggle CSVs placed here, or synthetic fallback
├── documents/             # project_presentation.pdf (use-case deck with screenshots)
├── notebooks/             # eda.ipynb / eda.py + notebooks/figures/*.png (committed)
├── app/                   # streamlit_app.py - the multi-tab UI (deviation, see below)
├── src/
│   ├── data/              # loader.py, preprocessor.py, synthetic.py
│   ├── ml/                # train.py, predict.py, evaluate.py, explain.py (SHAP - deviation)
│   ├── rules/             # rule_engine.py (deviation, see below)
│   ├── talk_to_data/      # prompt_templates.py, nl_to_sql.py, query_runner.py
│   └── utils/             # config.py, logger.py, helpers.py, docker_utils.py
├── sql/                   # schema.sql + HomeCredit_columns_description.csv
├── models/                # model.joblib, preprocessor.joblib, metrics.json, rules.json (gitignored, regenerated on first run)
├── Dockerfile / docker-compose.yml / entrypoint.sh
├── requirements.txt / .env.example / .gitignore
└── README.md
```

**Deviations from the assignment's published folder tree** (3, all additive):
`src/ml/explain.py`, `src/rules/`, and `app/` — the published tree has no
slot for the SHAP explainability module, the rule-derivation module, or the
UI code, yet all three are separately required deliverables in the same
document. Every other path matches the tree exactly.
