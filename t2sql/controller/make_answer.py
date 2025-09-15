from t2sql.utils import logger
import pandas as pd
from t2sql.base import BaseText2SQLAgent


async def make_answer(query: str, agent: BaseText2SQLAgent) -> str:
    logger.info(f"[{agent._descriptor_folder}] Extracting sql for query: {query}...")

    sql, _ = await agent.make_sql(query,
                                  reasoning_model_sql=agent.config.get("model_sql"),
                                  reasoning_model_table=agent.config.get("model_table_selection"))

    logger.info(f"[{agent._descriptor_folder}] Generated sql: {sql}")

    return sql


async def run_sql(sql: str, agent: BaseText2SQLAgent) -> tuple[str, pd.DataFrame]:
    sql, _, data = await agent.execute_sql(sql)

    logger.info(f"[{agent._descriptor_folder}] Returning data for sql: {len(data)}...")
    return sql, data


async def run_all(query: str, agent: BaseText2SQLAgent) -> tuple[str, pd.DataFrame]:
    sql = await make_answer(query, agent)
    return await run_sql(sql, agent)
