# AI-Powered Credit Risk Intelligence Platform

An end-to-end AI engineering solution for **credit risk assessment** built using the **Home Credit Default Risk** dataset.

The platform combines machine learning, explainable AI, business-rule extraction, and an LLM-powered **Natural Language → SQL** interface in a single interactive application.

The project demonstrates the complete AI engineering lifecycle:

**Data → EDA → Feature Engineering → ML → Risk Scoring → Explainability → Business Rules → NL-to-SQL → UI → Docker**

---

## Overview

Financial institutions need credit decisions that are not only accurate, but also **explainable, auditable, and accessible to business users**.

This platform provides:

- Exploratory Data Analysis and business insights
- Loan default probability prediction
- Low / Medium / High risk classification
- SHAP-based prediction explanations
- Business-readable risk rules
- Natural-language querying of credit data
- SQL validation and hallucination controls
- Conversational query context
- Interactive Streamlit dashboard
- Dockerized deployment

---

## System Architecture

```text
                         Home Credit Dataset
                                 │
                  ┌──────────────┴──────────────┐
                  │                             │
                  ▼                             ▼
           ML Data Pipeline                 DuckDB
                  │                             │
           Preprocessing                        │
                  │                       NL → SQL Layer
                  ▼                             │
        ┌──────────────────┐             ┌──────┴───────┐
        │ Logistic         │             │ LLM Prompt   │
        │ Regression       │             │ Engineering  │
        │       +          │             └──────┬───────┘
        │ LightGBM         │                    │
        └────────┬─────────┘              SQL Validation
                 │                              │
          Risk Probability                     ▼
                 │                         DuckDB Query
        ┌────────┴─────────┐                    │
        │                  │                    ▼
        ▼                  ▼              Business Answer
   Risk Bands          SHAP / Rules              │
        │                  │                     │
        └──────────────────┴──────────┬──────────┘
                                     ▼
                              Streamlit UI
                                     │
                                     ▼
                              Docker Compose
```

### Key design decisions

- **Streamlit** provides a lightweight multi-section interface for the complete workflow.
- **DuckDB** provides fast analytical SQL without requiring a separate database server.
- **LightGBM** is used as the primary tabular ML model, with **Logistic Regression** as an interpretable baseline.
- **SHAP** provides global and applicant-level model explanations.
- **Google Gemini** supports free-form natural-language-to-SQL generation.
- **Docker Compose** provides reproducible one-command deployment.

---

## Features

### 1. Exploratory Data Analysis

The EDA module investigates:

- Dataset and target distribution
- Missing values and data quality
- Applicant demographics
- Income and credit characteristics
- External credit scores
- Employment characteristics
- Default behaviour across customer segments

Key business insights are presented through interactive visualizations.

---

### 2. Credit Risk Prediction

The ML pipeline predicts an applicant's probability of loan default.

Two models are evaluated:

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Logistic Regression | 0.750 | 0.235 |
| **LightGBM + Isotonic Calibration** | **0.763** | **0.246** |

The dataset is highly imbalanced, with approximately **8% default cases**, so model selection focuses on **ROC-AUC and PR-AUC rather than accuracy alone**.

The final model outputs:

```text
Default Probability
        ↓
   Risk Score
        ↓
┌───────┼────────┐
LOW   MEDIUM    HIGH
```

Current configurable risk bands:

- **Low:** probability < 0.20
- **Medium:** 0.20 – 0.50
- **High:** ≥ 0.50

These bands are intended for decision support and can be adjusted according to business risk appetite.

---

### 3. Explainable AI

The platform uses **SHAP TreeExplainer** to explain LightGBM predictions.

For an individual applicant, the application identifies:

- factors increasing predicted default risk
- factors decreasing predicted default risk
- global model feature importance

Technical model outputs are translated into more understandable risk explanations for business users.

---

### 4. Business Risk Rules

Model behaviour is converted into interpretable decision-support rules using:

- shallow surrogate decision trees
- observed default rates across feature ranges
- model-driven risk segmentation

Example structure:

```text
IF external credit score is low
AND additional high-risk characteristics are present
THEN flag the applicant for additional review.
```

The rules provide an interpretable approximation of model behaviour and are **not intended to replace the ML risk score or formal lending policy**.

---

## Talk-to-Data: Natural Language → SQL

The platform allows business users to query credit data using plain English.

Example:

```text
User:
What is the default rate by education level?

        ↓

LLM generates SQL

        ↓

SQL Validator

        ↓

DuckDB executes query

        ↓

Result converted into a business-readable answer
```

Example questions include:

- What is the overall default rate?
- Which education group has the highest default rate?
- How does default rate vary across age groups?
- What is the average income of applicants?
- How many applicants own both a car and a house?

The generated SQL and result table are displayed in the UI for transparency.

---

## Prompt Engineering & Hallucination Control

Because LLM-generated SQL can be unreliable, the Talk-to-Data module includes several safeguards.

### Schema grounding

The LLM receives a curated database schema rather than unrestricted database context.

### SQL validation

Generated queries are checked before execution.

Only approved read-only queries are permitted.

Operations such as:

```sql
DROP
DELETE
UPDATE
INSERT
ALTER
```

are rejected.

Queries are restricted to approved tables and result sizes are limited.

### Verified query patterns

Frequently used analytical questions are mapped to pre-validated SQL templates, reducing unnecessary LLM calls and improving reliability.

### Token optimization

Only relevant schema information and limited conversational context are passed to the LLM.

### Conversation memory

Recent question/SQL pairs are retained so follow-up questions such as:

> "What about women only?"

can be interpreted using the previous analytical context.

---

## Application

The Streamlit application contains five main sections:

| Section | Purpose |
|---|---|
| 📊 **EDA** | Dataset exploration and business insights |
| 🎯 **Risk Scoring** | Applicant default probability and risk band |
| 🔬 **Explainability** | SHAP-based model explanations |
| ⚖️ **Business Rules** | Interpretable model-derived risk rules |
| 💬 **Ask the Data** | Natural-language querying using NL → SQL |

---

## Project Structure

```text
credit_risk_platform/
│
├── app/
│   ├── streamlit_app.py
│   ├── data.py
│   ├── theme.py
│   ├── styles.py
│   ├── ui.py
│   ├── charts.py
│   └── sections/
│
├── data/
│   └── Home Credit dataset
│
├── documents/
│   └── project presentation
│
├── notebooks/
│   ├── eda.py
│   ├── eda.ipynb
│   └── figures/
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   ├── preprocessor.py
│   │   └── synthetic.py
│   │
│   ├── ml/
│   │   ├── train.py
│   │   ├── predict.py
│   │   ├── evaluate.py
│   │   └── explain.py
│   │
│   ├── rules/
│   │   └── rule_engine.py
│   │
│   ├── talk_to_data/
│   │   ├── prompt_templates.py
│   │   ├── nl_to_sql.py
│   │   └── query_runner.py
│   │
│   └── utils/
│       ├── config.py
│       ├── logger.py
│       ├── helpers.py
│       └── docker_utils.py
│
├── sql/
│   ├── schema.sql
│   └── HomeCredit_columns_description.csv
│
├── models/
│
├── tests/
│
├── Dockerfile
├── docker-compose.yml
├── entrypoint.sh
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Getting Started

### Prerequisites

Either:

- Docker + Docker Compose

or:

- Python 3.11+

### 1. Clone the repository

```bash
git clone https://github.com/jinishar/credit_risk.git
cd credit_risk
```

### 2. Add the dataset

Download the **Home Credit Default Risk** dataset from Kaggle.

Place the required CSV files inside:

```text
data/
```

The dataset is intentionally excluded from Git.

The application also contains a synthetic-data fallback for demonstration and development when the original dataset is unavailable.

### 3. Configure environment variables

Copy:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Add a Gemini API key to enable free-form Talk-to-Data queries:

```env
GEMINI_API_KEY=your_api_key
```

Do **not** commit `.env`.

---

## Run with Docker

The recommended approach is:

```bash
docker compose up --build
```

Then open:

```text
http://localhost:8501
```

The initial run may train the model and generate required artifacts. Subsequent runs reuse the saved artifacts.

---

## Run Locally

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python -m src.ml.train
```

Generate business rules:

```bash
python -m src.rules.rule_engine
```

Start the application:

```bash
streamlit run app/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Covers the UI helpers, section imports, and the calibrated risk score.

---

## Model Evaluation

The final LightGBM model achieved:

| Metric | Validation Result |
|---|---:|
| ROC-AUC | **0.763** |
| PR-AUC | **0.246** |
| Default prevalence | ~8.07% |

ROC-AUC and PR-AUC are emphasized because accuracy can be misleading for an imbalanced default-prediction problem.

The model also applies probability calibration so the resulting scores can be interpreted more meaningfully as risk probabilities.

---

## Key EDA Findings

Analysis of the Home Credit application dataset highlighted several patterns:

1. **External credit scores are among the strongest predictors of default risk.**
2. **Default risk generally decreases with applicant age.**
3. **Education groups show meaningful differences in observed default rates.**
4. **Credit-to-income ratio alone provides limited separation and becomes more useful when combined with other risk indicators.**
5. **Family-status segments show different observed default behaviour.**
6. The dataset contains substantial missingness, requiring explicit handling during preprocessing.

These findings are exploratory associations and should not be interpreted as causal relationships.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Data Processing | Pandas / NumPy |
| Analytical Database | DuckDB |
| ML | Scikit-learn / LightGBM |
| Explainability | SHAP |
| LLM | Google Gemini |
| SQL Validation | sqlparse |
| UI | Streamlit |
| Visualization | Plotly / Matplotlib |
| Deployment | Docker / Docker Compose |

---

## Known Limitations

The current implementation has several areas for improvement:

- The primary ML model currently focuses on the application dataset rather than incorporating all bureau and previous-loan history tables.
- Model calibration currently uses the evaluation fold; a dedicated calibration split would provide a cleaner evaluation.
- Hyperparameter optimization is intentionally limited.
- Free-form NL-to-SQL queries require an external LLM API key.
- Model-derived business rules approximate model behaviour and should not be treated as formal lending policy.
- The system is an analytical prototype and has not undergone production banking validation, fairness auditing, or regulatory approval.

---

## Future Improvements

Potential extensions include:

- Aggregate bureau and previous-application history into the ML feature set
- Automated hyperparameter optimization
- Dedicated train / calibration / test splits
- Model drift monitoring
- Fairness and bias evaluation
- Stronger semantic schema retrieval for NL-to-SQL
- Additional SQL security controls
- API-based inference service
- Authentication and role-based access
- Production model monitoring

---

## Disclaimer

This project is an **AI engineering prototype for credit-risk analysis**.

Predictions, explanations, and business rules produced by the system are intended for demonstration and decision-support purposes only and should **not** be used as the sole basis for real-world lending decisions.

---

## Author

**Jinisha Leema Rosario**

AI / Machine Learning | Data Science | Generative AI

---

## Acknowledgements

Dataset: **Home Credit Default Risk**

Developed as part of the **NeoStats AI Engineer Use Case Assignment**.