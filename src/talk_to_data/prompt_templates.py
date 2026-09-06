"""Versioned prompt templates for the NL -> SQL talk-to-data chatbot.

Token-optimization approach (documented in README): the LLM is grounded with
a curated ~25-column schema summary rather than the full 121-column
dictionary, and conversation history sent back to the model is a short list
of prior (question, sql) pairs -- not raw result tables -- capped at the last
4 turns. This keeps each call's input small and bounded regardless of how
long the conversation runs.
"""
from __future__ import annotations

from src.utils.config import APPLICATIONS_TABLE

PROMPT_VERSION = "v1"

# Curated, token-efficient schema context (full detail lives in sql/schema.sql
# and sql/HomeCredit_columns_description.csv, but only the columns the
# chatbot actually needs are sent on every call).
SCHEMA_CONTEXT = f"""
Table: {APPLICATIONS_TABLE} (one row per loan applicant, ~307K rows on the real dataset)
Key columns:
  SK_ID_CURR                INTEGER  - applicant id
  TARGET                    INTEGER  - 1 = defaulted, 0 = repaid
  NAME_CONTRACT_TYPE        TEXT     - 'Cash loans' or 'Revolving loans'
  CODE_GENDER                TEXT    - 'M' or 'F'
  FLAG_OWN_CAR                TEXT   - 'Y' or 'N'
  FLAG_OWN_REALTY             TEXT   - 'Y' or 'N'
  CNT_CHILDREN                INTEGER
  AMT_INCOME_TOTAL            DOUBLE  - annual income
  AMT_CREDIT                  DOUBLE  - requested credit amount
  AMT_ANNUITY                 DOUBLE  - loan annuity (monthly payment)
  NAME_INCOME_TYPE            TEXT    - e.g. Working, Pensioner, Unemployed
  NAME_EDUCATION_TYPE         TEXT    - e.g. Higher education, Secondary
  NAME_FAMILY_STATUS          TEXT    - e.g. Married, Single / not married
  NAME_HOUSING_TYPE           TEXT    - e.g. House / apartment, Rented apartment
  DAYS_BIRTH                  INTEGER - negative; age_years = -DAYS_BIRTH/365.25
  DAYS_EMPLOYED                INTEGER - negative; 365243 = anomaly meaning "not employed"
  OCCUPATION_TYPE              TEXT
  ORGANIZATION_TYPE            TEXT
  EXT_SOURCE_1/2/3              DOUBLE - normalized external credit bureau scores (0-1, higher = safer)
""".strip()

GUARDRAILS = f"""
Rules you MUST follow when writing SQL:
1. Output ONLY a single DuckDB-compatible SELECT statement, nothing else - no
   prose, no markdown fences, no explanation.
2. The only table you may reference is `{APPLICATIONS_TABLE}`.
3. Never write INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, ATTACH, COPY,
   PRAGMA, TRUNCATE, GRANT, or multiple statements separated by `;`.
4. If the question cannot be answered from the columns above, write a SELECT
   that returns the closest reasonable approximation using only those
   columns - never invent a column name that isn't listed.
5. Always alias aggregate columns with a readable name (e.g. AS default_rate).
""".strip()

FEW_SHOT_EXAMPLES = [
    ("What is the overall default rate?",
     f"SELECT AVG(TARGET) * 100 AS default_rate_pct FROM {APPLICATIONS_TABLE}"),
    ("What is the average income of applicants who defaulted vs did not?",
     f"SELECT TARGET, AVG(AMT_INCOME_TOTAL) AS avg_income FROM {APPLICATIONS_TABLE} GROUP BY TARGET"),
    ("Show default rate by education level",
     f"SELECT NAME_EDUCATION_TYPE, AVG(TARGET) * 100 AS default_rate_pct, COUNT(*) AS n "
     f"FROM {APPLICATIONS_TABLE} GROUP BY NAME_EDUCATION_TYPE ORDER BY default_rate_pct DESC"),
    ("How many applicants own both a car and a house?",
     f"SELECT COUNT(*) AS n_applicants FROM {APPLICATIONS_TABLE} "
     f"WHERE FLAG_OWN_CAR = 'Y' AND FLAG_OWN_REALTY = 'Y'"),
    ("What are the top 5 organization types by number of applicants?",
     f"SELECT ORGANIZATION_TYPE, COUNT(*) AS n_applicants FROM {APPLICATIONS_TABLE} "
     f"GROUP BY ORGANIZATION_TYPE ORDER BY n_applicants DESC LIMIT 5"),
]

# The >=5 guaranteed-working query patterns required by the spec. Matched
# deterministically by src.talk_to_data.nl_to_sql before ever calling the
# LLM, so the demo always has a reliable, hallucination-free fallback for
# these even without a live Gemini key.
CANONICAL_PATTERNS: list[dict] = [
    {
        "keywords": ["overall default rate", "default rate overall", "what is the default rate", "how many default"],
        "question": "What is the overall default rate?",
        "sql": f"SELECT AVG(TARGET) * 100 AS default_rate_pct, COUNT(*) AS n_applicants FROM {APPLICATIONS_TABLE}",
    },
    {
        "keywords": ["average income", "income of applicants who defaulted", "income by target", "income vs default"],
        "question": "What is the average income of applicants who defaulted vs did not?",
        "sql": f"SELECT TARGET, AVG(AMT_INCOME_TOTAL) AS avg_income, COUNT(*) AS n "
               f"FROM {APPLICATIONS_TABLE} GROUP BY TARGET ORDER BY TARGET",
    },
    {
        "keywords": ["default rate by education", "education level", "education and default"],
        "question": "Show default rate by education level",
        "sql": f"SELECT NAME_EDUCATION_TYPE, AVG(TARGET) * 100 AS default_rate_pct, COUNT(*) AS n "
               f"FROM {APPLICATIONS_TABLE} GROUP BY NAME_EDUCATION_TYPE ORDER BY default_rate_pct DESC",
    },
    {
        "keywords": ["own a car and a house", "own car and realty", "car and house", "own both"],
        "question": "How many applicants own both a car and a house?",
        "sql": f"SELECT COUNT(*) AS n_applicants FROM {APPLICATIONS_TABLE} "
               f"WHERE FLAG_OWN_CAR = 'Y' AND FLAG_OWN_REALTY = 'Y'",
    },
    {
        "keywords": ["top 5 organization", "top organization types", "organization type by number", "most common organization"],
        "question": "What are the top 5 organization types by number of applicants?",
        "sql": f"SELECT ORGANIZATION_TYPE, COUNT(*) AS n_applicants FROM {APPLICATIONS_TABLE} "
               f"GROUP BY ORGANIZATION_TYPE ORDER BY n_applicants DESC LIMIT 5",
    },
]


def build_system_prompt() -> str:
    examples = "\n\n".join(f"Q: {q}\nSQL: {sql}" for q, sql in FEW_SHOT_EXAMPLES)
    return (
        f"You are a SQL analyst for a credit risk database. Convert natural language "
        f"questions about loan applicants into DuckDB SQL queries.\n\n"
        f"{SCHEMA_CONTEXT}\n\n{GUARDRAILS}\n\nExamples:\n{examples}"
    )


def build_history_context(history: list[dict], max_turns: int = 4) -> str:
    """Compact (question, sql) history for follow-up questions - no raw result data."""
    if not history:
        return ""
    recent = history[-max_turns:]
    lines = [f"Q: {turn['question']}\nSQL: {turn['sql']}" for turn in recent]
    return "Conversation so far:\n" + "\n\n".join(lines)


def build_narration_prompt(question: str, sql: str, result_preview: str) -> str:
    return (
        "You are a credit-risk business analyst. Explain the following SQL query "
        "result in 2-4 plain-English sentences for a non-technical bank manager. "
        "Do not mention SQL or column internals - speak in business terms. "
        "Write money amounts without a '$' sign (e.g. 'USD 169,078' or "
        "'169,078'). If the result is empty, say so plainly rather than "
        "inventing numbers.\n\n"
        f"Question: {question}\nSQL used: {sql}\nResult:\n{result_preview}"
    )
