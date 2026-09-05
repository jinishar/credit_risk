"""Natural language -> validated SQL, backed by Gemini with a deterministic fallback.

Hallucination / safety controls (documented in README):
  1. The >=5 canonical query patterns are matched by keyword before ever
     calling the LLM - these always return correct, pre-verified SQL with
     zero model risk.
  2. Every LLM-generated SQL string is parsed and validated before execution:
     single statement, SELECT-only, references only the whitelisted
     `applications` table, no DDL/DML/administrative keywords.
  3. If validation fails, the engine raises rather than silently executing
     unsafe or malformed SQL - the UI surfaces this as "couldn't safely
     answer that" instead of guessing.
"""
from __future__ import annotations

import re

import sqlparse

from src.talk_to_data.prompt_templates import (
    CANONICAL_PATTERNS,
    build_history_context,
    build_system_prompt,
)
from src.utils.config import APPLICATIONS_TABLE, GEMINI_API_KEY, GEMINI_MODEL
from src.utils.logger import get_logger

log = get_logger(__name__)

_FORBIDDEN_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "ATTACH", "DETACH",
    "COPY", "PRAGMA", "EXPORT", "IMPORT", "TRUNCATE", "GRANT", "REVOKE", "CALL",
    "EXEC", "EXECUTE", "REPLACE", "MERGE", "VACUUM", "SET",
}
_SQL_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


class SQLValidationError(ValueError):
    pass


def match_canonical(question: str) -> tuple[str, str] | None:
    """Returns (canonical_question, sql) if `question` matches one of the 5
    guaranteed patterns, else None."""
    q = question.lower().strip()
    for pattern in CANONICAL_PATTERNS:
        if any(kw in q for kw in pattern["keywords"]):
            return pattern["question"], pattern["sql"]
    return None


def extract_sql(raw_text: str) -> str:
    fenced = _SQL_FENCE_RE.search(raw_text)
    sql = fenced.group(1) if fenced else raw_text
    return sql.strip().rstrip(";").strip()


def validate_sql(sql: str) -> str:
    """Raises SQLValidationError if `sql` isn't a single safe SELECT on the
    whitelisted table; otherwise returns the (trimmed) validated SQL."""
    if not sql or not sql.strip():
        raise SQLValidationError("Empty SQL.")

    statements = [s for s in sqlparse.parse(sql) if s.tokens]
    if len(statements) != 1:
        raise SQLValidationError(f"Expected exactly one SQL statement, got {len(statements)}.")

    statement = statements[0]
    stmt_type = statement.get_type()
    if stmt_type != "SELECT":
        raise SQLValidationError(f"Only SELECT statements are allowed, got '{stmt_type}'.")

    upper_sql = sql.upper()
    for keyword in _FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", upper_sql):
            raise SQLValidationError(f"Forbidden keyword detected: {keyword}")

    tables = _referenced_tables(statement)
    disallowed = tables - {APPLICATIONS_TABLE.lower()}
    if disallowed:
        raise SQLValidationError(f"Query references non-whitelisted table(s): {disallowed}")

    if ";" in sql.strip().rstrip(";"):
        raise SQLValidationError("Multiple statements (stray ';') are not allowed.")

    return sql.strip()


def _referenced_tables(statement: sqlparse.sql.Statement) -> set[str]:
    tables: set[str] = set()
    tokens = list(statement.flatten())
    for i, token in enumerate(tokens):
        if token.ttype is sqlparse.tokens.Keyword and token.value.upper() in ("FROM", "JOIN"):
            for nxt in tokens[i + 1:]:
                if nxt.is_whitespace:
                    continue
                if nxt.ttype in (sqlparse.tokens.Name, None) or nxt.ttype is sqlparse.tokens.Literal.String.Symbol:
                    tables.add(nxt.value.strip('"').lower())
                break
    return tables


class NLToSQLEngine:
    def __init__(self) -> None:
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not GEMINI_API_KEY:
                raise RuntimeError(
                    "GEMINI_API_KEY is not set. Add it to your .env file to enable "
                    "free-form questions (the 5 canonical questions still work without it)."
                )
            from google import genai

            self._client = genai.Client(api_key=GEMINI_API_KEY)
        return self._client

    def generate_sql(self, question: str, history: list[dict] | None = None) -> tuple[str, bool]:
        """Returns (validated_sql, was_canonical). Tries the canonical matcher first."""
        canonical = match_canonical(question)
        if canonical is not None:
            _, sql = canonical
            return validate_sql(sql), True

        history_context = build_history_context(history or [])
        prompt = "\n\n".join(filter(None, [history_context, f"Q: {question}\nSQL:"]))

        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={"system_instruction": build_system_prompt(), "temperature": 0.1},
        )
        sql = extract_sql(response.text or "")
        return validate_sql(sql), False


if __name__ == "__main__":
    engine = NLToSQLEngine()
    for q in [
        "What is the overall default rate?",
        "Show default rate by education level",
        "How many applicants own both a car and a house?",
    ]:
        sql, is_canonical = engine.generate_sql(q)
        print(f"[{'canonical' if is_canonical else 'llm'}] {q}\n  -> {sql}\n")

    try:
        validate_sql("DROP TABLE applications")
    except SQLValidationError as e:
        print("Correctly rejected malicious SQL:", e)
