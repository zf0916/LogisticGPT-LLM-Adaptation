from __future__ import annotations
import time, os, asyncio
from typing import Literal, Any, List, Optional, Dict
import numpy as np
from string import Template

from t2sql.agent import Text2SQLAgent
from t2sql.base import TablesListLLM  # used by _extract_with_openai
from t2sql.utils import parse_code, logger, parse_json
try:
    import asyncpg  # for precise error classes if available
except Exception:
    asyncpg = None
import re

class MeasurableText2SQLAgent(Text2SQLAgent):
    """
    Extends Text2SQLAgent with:
      - Per-bucket usage accounting: generation vs table_selection
      - Optional quiet SQL execution (no giant stack traces)
      - last_usage / last_latency_ms for the *final* SQL generation call
    """
    def __init__(self, config: dict, vector_store):
        super().__init__(config, vector_store)
        self.last_usage: Optional[Dict] = None
        self.last_latency_ms: Optional[int] = None
        self.last_usage_bucket: Optional[str] = None

        self.usage_totals: Dict[str, int] = {"generation": 0, "table_selection": 0, "other": 0}
        self.latency_totals_ms: Dict[str, int] = {"generation": 0, "table_selection": 0, "other": 0}

        self._quiet_sql_errors: bool = bool(self.config.get("quiet_sql_errors", True))
        self._max_fix_attempts: int = int(self.config.get("max_sql_fix_attempts", 2))
        # New toggles
        self._autofix_enabled: bool = bool(self.config.get("sql_autofix_enabled", True))
        self._autofix_blacklist: set[str] = set(self.config.get(
            "sql_autofix_blacklist", ["undefined_column", "undefined_table"]
        ))
        self.last_exec_error: Optional[Dict[str, str]] = None

    # --- strip ONLY chat/formatting tokens (no schema normalization) ---
    @staticmethod
    def _pre_sanitize_sql(sql: Optional[str]) -> Optional[str]:
        if sql is None:
            return None
        s = sql.strip()
        s = re.sub(r"^<\s*/?s>\s*", "", s, flags=re.IGNORECASE)   # <s>
        s = re.sub(r"</\s*s>\s*$", "", s, flags=re.IGNORECASE)     # </s>
        s = re.sub(r"^<\|im_start\|>\s*", "", s)                   # <|im_start|>
        s = re.sub(r"\s*<\|im_end\|>\s*$", "", s)                  # <|im_end|>
        s = re.sub(r"^```(?:sql|postgresql)?\s*", "", s, flags=re.IGNORECASE)  # ```sql
        s = re.sub(r"\s*```$", "", s)                              # ```
        s = s.strip("` \n\t\r")                                    # stray backticks
        return s

    # ---------- usage tracking helper ----------
    async def _acompletion_track(self, bucket: str, **kwargs):
        t0 = time.time()
        res = await self._router.acompletion(**kwargs)
        t1 = time.time()

        # pull usage if provided by provider/Router
        def _u(x):
            if hasattr(x, "usage") and x.usage:
                u = x.usage
                return {
                    "prompt_tokens": getattr(u, "prompt_tokens", None) or (u.get("prompt_tokens") if isinstance(u, dict) else None),
                    "completion_tokens": getattr(u, "completion_tokens", None) or (u.get("completion_tokens") if isinstance(u, dict) else None),
                    "total_tokens": getattr(u, "total_tokens", None) or (u.get("total_tokens") if isinstance(u, dict) else None),
                }
            return None

        usage = _u(res)
        if usage and isinstance(usage.get("total_tokens"), (int, float)):
            self.usage_totals[bucket] = self.usage_totals.get(bucket, 0) + int(usage["total_tokens"])
        self.latency_totals_ms[bucket] = self.latency_totals_ms.get(bucket, 0) + int((t1 - t0) * 1000)

        # keep "last call" markers for convenience; set bucket, too
        self.last_usage = usage
        self.last_latency_ms = int((t1 - t0) * 1000)
        self.last_usage_bucket = bucket
        return res

    # ---------- OVERRIDES to capture tokens by bucket ----------

    # Final SQL generation -> bucket: "generation"
    async def generate_sql(
        self,
        question: str,
        sql_examples: str,
        relevant_tables: str,
        tables_to_sql: list[str],
        sql_write_template: Template,
        sql_reasoning: bool = True,
        reasoning_model: str = "o3-mini",
        reasoning_effort: Literal["low", "medium", "high"] = "medium",
    ) -> str:
        if sql_reasoning:
            logger.info(f".GENERATE SQL: reasoning {sql_reasoning}, model:{reasoning_model}")
        else:
            logger.info(f".GENERATE SQL: reasoning {sql_reasoning}, model: o1")

        request = sql_write_template.substitute({
            "question": question,
            "schema": self._schema,
            "instructions": self.get_prompt_string("DEFAULT_SQL_INSTRUCTIONS"),
            "tables": str(tables_to_sql),
        })
        messages = [
            {"role": "system", "content": f"You are a experienced Data Engineer. Your qualification is SQL code writing ({self._dialect} SQL)"},
            {"role": "system", "content": f"** EXAMPLES **\n\n{sql_examples}"},
            {"role": "system", "content": f"** DATASETS **\n\n{relevant_tables}"},
            {"role": "system", "content": f"** Business Rules **\n\n{self.business_rules_string}"},
            {"role": "user", "content": request},
        ]

        if sql_reasoning:
            messages_t = [{"role": "user", "content": m["content"]} for m in messages]
            messages_t.append({"role": "user", "content": "Format sql as markdown code block with sql syntax."})
            try:
                ai_msg = await self._acompletion_track("generation",
                    model=reasoning_model, messages=messages_t, reasoning_effort=reasoning_effort
                )
            except:
                ai_msg = await self._acompletion_track("generation",
                    model=reasoning_model, messages=messages_t
                )
        else:
            r = 1 if reasoning_model == "simple" else 5
            ai_msg = await self._acompletion_track("generation",
                messages=messages, model=self.default_model, n=r, temperature=0
            )

        sqls: List[str] = []
        for ch in ai_msg.choices:
            try:
                sqls.append(self._pre_sanitize_sql(parse_code(ch.message.content)))
            except Exception as e:
                logger.error(f"Error parsing sql results: {str(e)}")
        sqls, counts = np.unique(sqls, return_counts=True)
        return self._pre_sanitize_sql(sqls[np.argmax(counts)])

    # Table-selection paths -> bucket: "table_selection"
    async def _extract_with_openai(
        self,
        messages: list[dict],
        model_type: str,
        reasoning_effort: Literal["low", "medium", "high"] = "medium",
    ) -> list[str]:
        if model_type == "o1":
            kwargs = {"reasoning_effort": "low"}
        elif model_type == "o1-mini":
            kwargs = {}
        else:
            kwargs = {"reasoning_effort": reasoning_effort}

        response = await self._acompletion_track("table_selection",
            model=model_type, messages=messages, response_format=TablesListLLM, **kwargs
        )
        result = TablesListLLM(**parse_json(response.choices[0].message.content))
        return result.tables

    async def get_tables_from_business_rules(
        self,
        question: str,
        model: str = "o3-mini",
        reasoning_effort: Literal["low", "medium", "high"] = "medium",
    ) -> list[str]:
        try:
            messages = [
                {"role": "system", "content": f"""Business Rules:\n\n{self._business_rules}"""},
                {"role": "user", "content": self.get_prompt("EXTRACT_TABLES_BR").substitute({"question": question})},
            ]
            try:
                ai_res = await self._acompletion_track("table_selection",
                    model=model, messages=messages, reasoning_effort=reasoning_effort, response_format=TablesListLLM
                )
            except:
                ai_res = await self._acompletion_track("table_selection",
                    model=model, messages=messages, response_format=TablesListLLM
                )
            result = TablesListLLM(**parse_json(ai_res.choices[0].message.content))
        except Exception as e:
            result = TablesListLLM()
            logger.error(f"Error extracting relevant tables based on business rules: {str(e)}")
        return result.tables

    # ---------- QUIETER SQL execution (no giant traces) ----------
    async def execute_sql(self, sql: str):
        to_fix = True
        cnt = 0
        is_success = False
        import pandas as pd
        data = pd.DataFrame()

        max_attempts = max(0, int(self._max_fix_attempts if self._autofix_enabled else 0))
        timeout_s = int(self.config.get("sql_exec_timeout_s", 30))  # default 30s

        def _classify_sql_error(e: Exception) -> str:
            s = (repr(e) or str(e) or "").lower()
            if asyncpg and isinstance(e, getattr(asyncpg.exceptions, "UndefinedColumnError", tuple())):
                return "undefined_column"
            if "undefinedcolumnerror" in s or "undefined column" in s:
                return "undefined_column"
            if asyncpg and isinstance(e, getattr(asyncpg.exceptions, "UndefinedTableError", tuple())):
                return "undefined_table"
            if "undefinedtableerror" in s or ("relation" in s and "does not exist" in s):
                return "undefined_table"
            if "permission denied" in s:
                return "permission_denied"
            if "syntax" in s:
                return "syntax_error"
            return "other"

        # pre-sanitize once to avoid trivial '<s>' / fences errors
        sql = self._pre_sanitize_sql(sql)
        while to_fix:
            try:
                # hard timeout around the DB call
                data = await asyncio.wait_for(self._client.execute_query(sql), timeout=timeout_s)
                to_fix = False
                is_success = True
            except asyncio.TimeoutError:
                if self._quiet_sql_errors:
                    logger.warning(f"SQL timeout after {timeout_s}s")
                else:
                    logger.error(f"SQL timeout after {timeout_s}s")
                # do not try to "fix" a timed-out query repeatedly
                to_fix = False
                data = pd.DataFrame()
                is_success = False
            except Exception as e:
                category = _classify_sql_error(e)
                self.last_exec_error = {"category": category, "message": str(e)}
                if self._quiet_sql_errors:
                    logger.warning(f"SQL error ({category}): {e.__class__.__name__}: {str(e)}")
                else:
                    # fallback to noisier behavior if you want
                    logger.error(f"Error executing query: {str(e)}")

                # Skip LLM auto-fix for blacklisted fatal categories or if autofix disabled
                if (category in self._autofix_blacklist) or (max_attempts <= 0) or (cnt >= max_attempts):
                    if category in self._autofix_blacklist:
                        logger.info(f"Skip auto-fix for fatal category: {category}")
                    to_fix = False
                    data = pd.DataFrame()
                else:
                    # attempt LLM-driven fix
                    try:
                        import traceback as _tb
                        error_text = _tb.format_exception(e)[-1]
                    except Exception:
                        error_text = str(e)
                    sql = self._pre_sanitize_sql(await self._fix_sql(sql, error_text))
                    cnt += 1
                    data = pd.DataFrame()

        return sql, is_success, data
