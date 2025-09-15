import numpy as np
import traceback
from t2sql.utils import logger
from typing import Literal
from t2sql.vectordb.chromadb import VectorStore, ChromaDB
from t2sql.ingestors.text_document_ingestor import TextIngestion
from t2sql.utils import get_config


class Text2SQLAgent(TextIngestion):
    def __init__(self, config: dict, vector_store: VectorStore):
        super().__init__(config, vector_store)

    # TODO: break down no methods for cleanliness
    async def make_sql(
        self,
        question: str,
        sql_reasoning: bool = True,
        reasoning_model_sql: str = "o3-mini",
        is_reranking: bool = True,
        seek_table_reasoning: bool = True,
        reasoning_model_table: str = "o3-mini",
        reasoning_effort_sql: Literal["low", "medium", "high"] = "medium",
        reasoning_effort_table: Literal["low", "medium", "high"] = "medium",
    ) -> tuple[str, str]:
        """
        Generates SQL query for a given question using various reasoning approaches and example matching.
        This method implements a multi-step process to generate the most accurate SQL query possible.

        Args:
            question (str): The natural language question to generate SQL for
            sql_reasoning (bool): Whether to use advanced SQL reasoning capabilities
            reasoning_model_sql (str): The model to use for SQL generation reasoning (default: RM_OPENAI_O3_MINI)
            is_reranking (bool): Whether to rerank similar questions for better matching
            seek_table_reasoning (bool): Whether to use advanced table selection reasoning
            reasoning_model_table (str): The model to use for table selection reasoning
            reasoning_effort_sql (Literal["low", "medium", "high"]): Level of effort for SQL reasoning
            reasoning_effort_table (Literal["low", "medium", "high"]): Level of effort for table reasoning

        Returns:
            tuple[str, str]: A tuple containing:
                - The generated SQL query as a string
                - The step identifier indicating which method was used to generate the query

        Process:
            1. First attempts to find exact matches from existing examples
            2. If no exact match, normalizes the question and tries matching again
            3. If still no match, analyzes entities and searches for similar questions
            4. If needed, performs full table analysis and SQL generation

        Notes:
            - The method uses progressively more complex approaches until it finds a suitable solution
            - Returns early if it finds high-confidence matches in earlier steps
            - Uses logging to track the progress through different stages
        """

        tables_to_sql = []
        logger.info(".EXTRACTING EXAMPLES")
        logger.info("..Exact Examples for user's request")
        sqls_max = []
        sqls, _, best_flag, list_tables, _ = await self.get_similar_question_sql(
            question, init_question=question, best_score=1e-5, best_only=True
        )
        if best_flag:
            logger.info("..Found example with best match.")
            return sqls[0]["sql"], "QUESTION EXACT MATCHING"

        step = "UNKNOWN"
        structure = await self.normalize_and_structure(question)
        logger.info(f"..Normalized question to: {structure.main_clause}.")
        tables_question = []

        if len(structure.data_source) > 0:
            for ds in structure.data_source:
                tables_question.append(ds.source)

        logger.info("..Exact Examples for normalized question")
        sqls, _, best_flag, list_tables, _ = await self.get_similar_question_sql(
            structure.normalized_question,
            init_question=question,
            best_score=0.01,
            best_only=True,
        )
        write_code_template = self.get_prompt("WRITE_CODE_EXAMPLES")
        if best_flag and len(tables_question) == 0:
            logger.info("..Found example with best match.")
            return sqls[0]["sql"], "NORMALIZED QUESTION EXACT MATCHING"
        elif best_flag and len(tables_question) > 0:
            tables_question = np.unique(tables_question + list_tables)
            relevant_tables, _tables = await self.return_table_docs(tables_question)
            sqls_add = []
        else:
            norm_question = structure.normalized_question
            logger.info("..Search examples for expected entities")
            (
                sqls,
                sqls_max,
                best_flag,
                list_tables,
                list_tables_max,
            ) = await self.get_similar_question_sql(
                structure.main_clause,
                init_question=question,
                add_question=norm_question,
                break_if_close=False,
                is_reranking=is_reranking,
            )

            if len(sqls) > 0:
                tbls = await self.get_tables_from_business_rules(question=norm_question)
                docs_to_sql, _tables = await self.return_table_docs(
                    np.unique(list_tables + tbls + tables_question)
                )
                relevant_tables = docs_to_sql
                sqls_add = []
                step = "MAIN CLAUSE BASED RESULT"
                logger.info(f"..Found documents with best match: {_tables}.")
            else:
                sqls_add = []
                relevant_tables, _tables = await self.get_related_tables(
                    norm_question=structure.normalized_question,
                    main_clause=structure.main_clause,
                    concepts=structure.requested_entities,
                    list_tables_max=tables_question,
                    seek_table_reasoning=seek_table_reasoning,
                    reasoning_model=reasoning_model_table,
                    reasoning_effort=reasoning_effort_table,
                )
                step = "DOCUMENTATION BASED RESULT"
                write_code_template = self.get_prompt("WRITE_CODE_DOCUMENTATION")
                tables_to_sql = _tables
                logger.info(f"..Found documents with best match: {_tables}.")

        if len(sqls_add) == 0:
            sqls_add = sqls_max

        sqls = sqls + sqls_add
        qws = []
        sql_res = []
        for sql in sqls:
            if sql["structure"]["init_question"] not in qws:
                qws.append(sql["structure"]["init_question"])
                sql_res.append(sql)

        sql_examples = ""
        for d in sql_res:
            qst = d["structure"]["init_question"]
            sql = d["sql"]
            sql_examples = (
                sql_examples
                + "\n\n###\n"
                + f"QUESTION:\n{qst}\nSQL that answers this question:\n<code>{sql}</code>"
            )

        logger.info("..Writing SQL")
        sql = await self.generate_sql(
            question,
            sql_examples,
            relevant_tables,
            tables_to_sql=tables_to_sql,
            sql_write_template=write_code_template,
            sql_reasoning=sql_reasoning,
            reasoning_model=reasoning_model_sql,
            reasoning_effort=reasoning_effort_sql,
        )

        return sql, step


def get_sql_agent(descriptor_base_path: str | None = None) -> Text2SQLAgent:
    try:
        config = get_config(descriptor_base_path)
        vector_store = ChromaDB(config)
        return Text2SQLAgent(config, vector_store)
    except Exception as e:
        logger.error(f"Cannot create Text2SQL Agent, error: {e}")
        traceback.print_exc()
        raise Exception("Cannot create agent, bad descriptor.")
