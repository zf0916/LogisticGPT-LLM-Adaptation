# text2sql/extensions/api.py
from __future__ import annotations
from typing import Tuple, Optional, Dict

from t2sql.base import BaseText2SQLAgent
from t2sql.utils import logger


async def make_answer_ex(query: str, agent: BaseText2SQLAgent) -> Tuple[str, str, Optional[Dict]]:
    """
    Returns (sql, step, usage_dict).
    usage_dict is populated when agent is MeasurableText2SQLAgent and the provider exposes usage.
    """
    logger.info(f"[{getattr(agent, '_descriptor_folder', '?')}] Extracting sql for query: {query}...")

    sql, step = await agent.make_sql(
        query,
        reasoning_model_sql=agent.config.get("model_sql"),
        reasoning_model_table=agent.config.get("model_table_selection"),
    )
    logger.info(f"[{getattr(agent, '_descriptor_folder', '?')}] Generated sql: {sql}")

    # Best-effort: measurable agent sets .last_usage
    usage = getattr(agent, "last_usage", None)
    return sql, step, usage