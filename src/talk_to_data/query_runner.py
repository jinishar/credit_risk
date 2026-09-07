"""Executes validated SQL on DuckDB and narrates the result in business language.

Conversation memory: `ChatSession` keeps a rolling list of prior
(question, sql) turns (see prompt_templates.build_history_context) so
follow-up questions like "and what about for women only?" can be resolved -
capped at the last 4 turns to bound token usage per call.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.data.loader import get_duckdb_connection
from src.talk_to_data.nl_to_sql import NLToSQLEngine, SQLValidationError
from src.talk_to_data.prompt_templates import build_narration_prompt
from src.utils.config import GEMINI_API_KEY, GEMINI_MODEL
from src.utils.logger import get_logger

log = get_logger(__name__)

MAX_ROWS_RETURNED = 200
MAX_ROWS_IN_NARRATION = 15


@dataclass
class ChatTurn:
    question: str
    sql: str
    is_canonical: bool
    result: pd.DataFrame
    narrative: str
    error: str | None = None


@dataclass
class ChatSession:
    """One conversation's state: engine, DB connection and rolling history.

    `connection` lets the caller pass a shared, already-loaded DuckDB
    connection (the Streamlit app caches one per server); when omitted a fresh
    one is opened, which is what the CLI and tests want.
    """
    engine: NLToSQLEngine = field(default_factory=NLToSQLEngine)
    history: list[dict] = field(default_factory=list)
    connection: object | None = None

    def __post_init__(self) -> None:
        self._con = self.connection if self.connection is not None else get_duckdb_connection()

    def ask(self, question: str) -> ChatTurn:
        try:
            sql, is_canonical = self.engine.generate_sql(question, self.history)
        except SQLValidationError as e:
            turn = ChatTurn(question, sql="", is_canonical=False, result=pd.DataFrame(),
                             narrative="I couldn't safely turn that into a query.", error=str(e))
            return turn
        except RuntimeError as e:
            turn = ChatTurn(question, sql="", is_canonical=False, result=pd.DataFrame(),
                             narrative=str(e), error=str(e))
            return turn

        try:
            df = self._run(sql)
        except Exception as e:  # DuckDB execution error (bad column name, etc.)
            log.warning("SQL execution failed: %s | sql=%s", e, sql)
            turn = ChatTurn(question, sql=sql, is_canonical=is_canonical, result=pd.DataFrame(),
                             narrative="That query didn't run successfully against the data.", error=str(e))
            return turn

        narrative = self._narrate(question, sql, df)
        self.history.append({"question": question, "sql": sql})
        return ChatTurn(question, sql, is_canonical, df, narrative)

    def _run(self, sql: str) -> pd.DataFrame:
        if "limit" not in sql.lower():
            sql = f"{sql} LIMIT {MAX_ROWS_RETURNED}"
        # A per-call cursor keeps queries thread-safe when the connection is
        # shared across Streamlit sessions.
        return self._con.cursor().execute(sql).fetchdf()

    def _narrate(self, question: str, sql: str, df: pd.DataFrame) -> str:
        if df.empty:
            return "That query returned no matching rows."
        if not GEMINI_API_KEY:
            return self._template_narrative(df)

        preview = df.head(MAX_ROWS_IN_NARRATION).to_string(index=False)
        try:
            response = self.engine.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=build_narration_prompt(question, sql, preview),
                config={"temperature": 0.2},
            )
            return (response.text or "").strip() or self._template_narrative(df)
        except Exception as e:  # narration failing shouldn't break the whole answer
            log.warning("Narration call failed, falling back to template: %s", e)
            return self._template_narrative(df)

    @staticmethod
    def _template_narrative(df: pd.DataFrame) -> str:
        if df.shape == (1, 1):
            return f"Result: {df.iloc[0, 0]}"
        return f"Returned {len(df)} row(s) across columns: {', '.join(df.columns)}."
