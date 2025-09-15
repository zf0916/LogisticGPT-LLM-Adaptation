from abc import abstractmethod, ABC
from string import Template
from t2sql.utils import logger
from t2sql.types import TrainingPlan, TrainingPlanItem
from litellm import Router
from sentence_transformers import CrossEncoder
from chromadb import QueryResult
from typing import Literal
from t2sql.utils import deterministic_uuid, load_prompts
import pandas as pd
import traceback
from t2sql.sql.client import DatabaseClientFactory
from t2sql.utils import parse_code
import numpy as np
from t2sql.vectordb.chromadb import VectorStore
from typing import Any
from t2sql.utils import parse_json, calculate_threshold
import json
from pydantic import BaseModel
import re

class ExpandQuestionLLM(BaseModel):
    chain_of_thoughts: str
    questions: list[str]


class TablesListLLM(BaseModel):
    tables: list[str]


class TableWhyLLM(BaseModel):
    table: str
    why: str


class TableWhyListLLM(BaseModel):
    tables: list[TableWhyLLM]


class Calculations(BaseModel):
    operation: str
    arguments: list[str]
    grouping: list[str]
    conditions: str


class DataSource(BaseModel):
    source: str
    columns: list[str]


class NormalizedStructureLLM(BaseModel):
    init_question: str | None = None
    tables: list[str] = []
    normalized_question: str
    requested_entities: str
    data_source: list[DataSource]
    calculations: list[Calculations]
    main_clause: str | None = None


class MainClauseLLM(BaseModel):
    main_clause: str | None = None
    details: str | None = None


MODEL_CROSS = CrossEncoder("cross-encoder/quora-roberta-large")


class BaseText2SQLAgent(ABC):
    _prompts: dict = {}
    _business_rules: list = []
    _vector_store: VectorStore | None = None

    default_sql_min_score: float = 0.06
    default_sql_max_score: float = 0.18
    default_sql_best_score: float = 0.03
    default_sql_min_score_table: float = 0.05
    default_sql_best_score_table: float = 0.01
    default_ddl_min_score: float = 1.3
    default_doc_min_score: float = 1.3
    default_sql_best_rerank_score: float = 0.05
    default_sql_min_rerank_score: float = 0.13
    default_sql_max_rerank_score: float = 0.4

    def __init__(self, config: dict, vector_store: VectorStore):
        self._config = config
        self._business_rules = config.get("business_rules", [])
        self._vector_store = vector_store
        self._prompts = load_prompts(config.get("descriptors_folder"))
        self._dialect = self._config.get("db").get("source")
        self._database = self._config.get("db").get("connection_config").get("database")
        self._schema = self._config.get("db").get("connection_config").get("schema")
        self._client = DatabaseClientFactory.create_client(**self._config.get("db"))
        self._descriptor_folder = config.get("descriptors_folder")
        self._router = Router(
            model_list=config["router_model_list"],
            num_retries=config[
                "router_default_num_retries"
            ],
            default_max_parallel_requests=config[
                "router_default_max_parallel_requests"
            ],
        )

    @property
    def default_model(self) -> str:
        # Use table/text model for everything except raw SQL generation/fixing
        return self._config["model_table_selection"]

    @property
    def model_sql(self) -> str:
        # Use only for SQL generation / SQL fixing
        return self._config["model_sql"]

    @property
    def config(self):
        return self._config

    @property
    def business_rules(self):
        return self._business_rules

    @property
    def business_rules_string(self) -> str:
        br_str = ""
        for k, br in enumerate(self._business_rules):
            br_str = br_str + f"{k + 1}. {br}\n"
        return br_str

    def get_prompt(self, key: str) -> Template:
        return Template(self._prompts[key])

    def get_prompt_string(self, key: str) -> str:
        return self._prompts[key]

    def refresh_business_rules(self, rules: list[str]) -> None:
        self._business_rules = rules

    def refresh_prompts(self) -> None:
        self._prompts = load_prompts(self._config.get("descriptors_folder"))

    @staticmethod
    def _add_add_question_sql_to_sql_result(
        query_result: QueryResult, query_result_add: QueryResult
    ) -> QueryResult:
        for id, doc, mtd, dst in zip(
            query_result_add["ids"][0],
            query_result_add["documents"][0],
            query_result_add["metadatas"][0],
            query_result_add["distances"][0],
        ):
            if id not in query_result["ids"][0]:
                query_result["ids"][0].append(id)
                query_result["documents"][0].append(doc)
                query_result["metadatas"][0].append(mtd)
                query_result["distances"][0].append(dst)
            else:
                min_dst = min(
                    dst, query_result["distances"][0][query_result["ids"][0] == id]
                )
                query_result["distances"][0][query_result["ids"][0] == id] = min_dst
        return query_result

    @staticmethod
    def _filter_by_distance_score_sql(
        result: list,
        query_result: QueryResult,
        min_score: float,
        best_score: float,
        break_if_close: bool = False,
        indexes: list | None = None,
    ) -> tuple[list, bool, list]:
        filtered = []
        best_example = False
        tables = []
        if not indexes:
            indexes = list(range(len(query_result["distances"][0])))

        for index, distance in zip(
            np.argsort(query_result["distances"][0]),
            np.sort(query_result["distances"][0]),
        ):
            if index in indexes:
                if distance < min_score:
                    tt = query_result["metadatas"][0][index].get("structure", {})
                    tt = json.loads(tt)
                    r = result[index]
                    r["structure"] = tt
                    filtered.append(r)
                    tt = tt.get("tables", [])
                    tables = tables + tt
                    tables = list(np.unique(tables))
                    if distance < best_score:
                        best_example = True
                        if break_if_close:
                            break

        return filtered, best_example, tables

    def _rerank(
        self,
        query_result: QueryResult,
        init_question: str,
        is_reranking: bool,
        break_if_close: bool,
    ) -> tuple[list, list, bool, list, list]:
        corpus = []
        for mtd, dst in zip(query_result["metadatas"][0], query_result["distances"][0]):
            if dst <= self.default_sql_max_score:
                corpus.append(json.loads(mtd["structure"])["init_question"])

        corpus_short, counts = np.unique(corpus, return_counts=True)
        indx = []
        for q in corpus_short:
            i = (np.array(corpus) == q).argmax()
            indx.append(i)

        result_max_rerank = []
        result_rerank = []
        tables_rerank = []
        tables_max_rerank = []
        best_flag = False

        if is_reranking and len(corpus_short) > 0:
            logger.info("...RERANKING")
            try:
                # TODO explore async version
                rank = MODEL_CROSS.rank(init_question, corpus_short)
                for r in rank:
                    query_result["distances"][0][indx[r["corpus_id"]]] = 1 - r["score"]

                result_rerank = self._vector_store.extract_documents(query_result)
                result_rerank, best_flag, tables_rerank = (
                    self._filter_by_distance_score_sql(
                        result_rerank,
                        query_result,
                        self.default_sql_min_rerank_score,
                        break_if_close=break_if_close,
                        best_score=self.default_sql_best_rerank_score,
                        indexes=indx,
                    )
                )

                result_max_rerank = self._vector_store.extract_documents(query_result)
                result_max_rerank, _, tables_max_rerank = (
                    self._filter_by_distance_score_sql(
                        result_max_rerank,
                        query_result,
                        self.default_sql_max_rerank_score,
                        break_if_close=break_if_close,
                        best_score=self.default_sql_best_rerank_score,
                        indexes=indx,
                    )
                )
            except Exception as e:
                logger.error(f"Error reranking in getting similar questions: {str(e)}")
                traceback.print_exc()

        return (
            result_rerank,
            result_max_rerank,
            best_flag,
            tables_rerank,
            tables_max_rerank,
        )

    async def get_similar_question_sql(
        self,
        question: str,
        init_question: str | None = None,
        add_question: str | None = None,
        best_score: float | None = None,
        best_only: bool = False,
        break_if_close: bool = True,
        is_reranking: bool = False,
    ) -> tuple[list, list, bool, list, list]:
        init_question = question if not init_question else init_question
        if not best_score:
            best_score = self.default_sql_best_score

        query_result = await self._vector_store.get_related_sql(question)
        if add_question:
            query_result_add = await self._vector_store.get_related_sql(add_question)
            query_result = self._add_add_question_sql_to_sql_result(
                query_result, query_result_add
            )

        result = self._vector_store.extract_documents(query_result)
        result, best_flag, tables = self._filter_by_distance_score_sql(
            result,
            query_result,
            self.default_sql_min_score,
            break_if_close=break_if_close,
            best_score=best_score,
        )
        result_max = self._vector_store.extract_documents(query_result)
        result_max, _, tables_max = self._filter_by_distance_score_sql(
            result_max,
            query_result,
            self.default_sql_max_score,
            break_if_close=break_if_close,
            best_score=best_score,
        )
        if (best_flag or best_only) and not is_reranking:
            return result, result_max, best_flag, tables, tables_max

        (
            result_rerank,
            result_max_rerank,
            best_flag,
            tables_rerank,
            tables_max_rerank,
        ) = self._rerank(query_result, init_question, is_reranking, break_if_close)

        if best_flag:
            return (
                result_rerank,
                result_max_rerank,
                best_flag,
                tables_rerank,
                tables_max_rerank,
            )

        result_max = result_max_rerank if len(result_max_rerank) != 0 else result_max
        result = result_rerank
        tables_max = tables_max_rerank if len(result_max_rerank) != 0 else tables_max
        tables = tables_rerank

        return result, result_max, best_flag, tables, tables_max

    async def _extract_tables_from_sql(self, sql_code: str) -> list[str]:
        request = self.get_prompt("EXTRACT_TABLES_CODE").substitute({"code": sql_code})

        messages = [
            {
                "role": "system",
                "content": "You are an experienced Text Processing Specialist",
            },
            {"role": "user", "content": request},
        ]

        ai_msg = await self._router.acompletion(
            messages=messages,
            model=self.default_model,
            response_format=TablesListLLM,
        )
        answer = TablesListLLM(**parse_json(ai_msg.choices[0].message.content))
        return answer.tables

    async def normalize_and_structure(
        self, question, sql=None
    ) -> NormalizedStructureLLM:
        sql_snippet = f"""\n\nSQL SNIPPET:\n{sql}""" if sql is not None else ""
        messages = [
            {
                "role": "system",
                "content": self.get_prompt_string("NORMALIZE_AND_STRUCTURE"),
            },
            {
                "role": "user",
                "content": f'''QUESTION:\n"{question}{sql_snippet}" Do not forget put all values to the conditions!''',
            },
        ]

        ai_msg = await self._router.acompletion(
            messages=messages,
            model=self.default_model,
            response_format=NormalizedStructureLLM,
        )

        result = NormalizedStructureLLM(**parse_json(ai_msg.choices[0].message.content))
        result.init_question = question
        if sql is not None:
            tables = await self._extract_tables_from_sql(sql)
            result.tables = tables

        request = self.get_prompt("MAIN_CLAUSE").substitute({"question": question})
        ai_msg = await self._router.acompletion(
            messages=[{"role": "user", "content": request}],
            model=self.default_model,
            response_format=MainClauseLLM,
        )
        answer = MainClauseLLM(**parse_json(ai_msg.choices[0].message.content))

        result.main_clause = (
            answer.main_clause
            if answer.main_clause is not None
            else result.requested_entities
        )
        return result

    async def get_tables_from_business_rules(
        self,
        question: str,
        model: str = "o3-mini",
        reasoning_effort: Literal["low", "medium", "high"] = "medium",
    ) -> list[str]:
        """
        Extract relevant tables based on business rules.

        Args:
            question (str): User's question
            model (str): Model name to use
            reasoning_effort (str): Level of reasoning effort ('low', 'medium', 'high')

        Returns:
            list: List of relevant table names
        """

        try:
            messages = [
                {
                    "role": "system",
                    "content": f"""Business Rules:\n\n{self._business_rules}""",
                },
                {
                    "role": "user",
                    "content": self.get_prompt("EXTRACT_TABLES_BR").substitute(
                        {"question": question}
                    ),
                },
            ]

            try:
                ai_res = await self._router.acompletion(
                    model=model,
                    messages=messages,
                    reasoning_effort=reasoning_effort,
                    response_format=TablesListLLM,
                )
            except:
                # Models that do not support reasoning_effort
                ai_res = await self._router.acompletion(
                    model=model, messages=messages, response_format=TablesListLLM
                )

            result = TablesListLLM(**parse_json(ai_res.choices[0].message.content))
        except Exception as e:
            result = TablesListLLM()
            logger.error(
                f"Error extracting relevant tables based on business rules: {str(e)}"
            )
            traceback.print_exc()

        return result.tables

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
        """
        Generate SQL query from natural language question.

        Args:
            question (str): Natural language question
            sql_examples (str): Examples of similar SQL queries
            relevant_tables (str): Relevant table documentation
            tables_to_sql (list): List of tables to use
            sql_write_template (str): Template for SQL generation
            sql_reasoning (bool): Whether to use reasoning
            reasoning_model (str): Model to use for reasoning
            reasoning_effort (str): Level of reasoning effort

        Returns:
            str: Generated SQL query
        """

        if sql_reasoning:
            logger.info(
                f".GENERATE SQL: reasoning {sql_reasoning}, model:{reasoning_model}"
            )
        else:
            logger.info(f".GENERATE SQL: reasoning {sql_reasoning}, model: o1")

        request = sql_write_template.substitute(
            {
                "question": question,
                "schema": self._schema,
                "instructions": self.get_prompt_string("DEFAULT_SQL_INSTRUCTIONS"),
                "tables": str(tables_to_sql),
            }
        )
        messages = [
            {
                "role": "system",
                "content": f"You are a experienced Data Engineer. Your qualification is SQL code writing ({self._dialect} SQL)",
            },
            {"role": "system", "content": f"** EXAMPLES **\n\n{sql_examples}"},
            {"role": "system", "content": f"** DATASETS **\n\n{relevant_tables}"},
            {
                "role": "system",
                "content": f"** Business Rules **\n\n{self.business_rules_string}",
            },
            {"role": "user", "content": request},
        ]

        if sql_reasoning:
            messages_t = []
            for m in messages:
                m["role"] = "user"
                messages_t.append(m)
            messages_t.append(
                {
                    "role": "user",
                    "content": "Format sql as markdown code block with sql syntax.",
                }
            )

            try:
                ai_msg = await self._router.acompletion(
                    model=reasoning_model,
                    messages=messages_t,
                    reasoning_effort=reasoning_effort,
                )
            except:
                # Models that do not support reasoning_effort
                ai_msg = await self._router.acompletion(
                    model=reasoning_model,
                    messages=messages_t,
                )
        else:
            r = 1 if reasoning_model == "simple" else 5
            ai_msg = await self._router.acompletion(
                messages=messages, model=self.model_sql, n=r, temperature=0
            )

        sqls = []
        for ch in ai_msg.choices:
            try:
                sqls.append(parse_code(ch.message.content))
            except Exception as e:
                logger.error(f"Error parsing sql results: {str(e)}")
                traceback.print_exc()

        sqls, counts = np.unique(sqls, return_counts=True)
        sql = sqls[np.argmax(counts)]
        return sql

    async def _fix_sql(self, sql: str, error: str) -> str:
        """
        Fix SQL query based on error message.

        Args:
            sql (str): Original SQL query
            error (str): Error message

        Returns:
            str: Fixed SQL query
        """
        request = self.get_prompt("FIX_CODE").substitute(
            {"error": error, "sql": sql, "dialect": self._dialect}
        )

        messages = [{"role": "user", "content": request}]

        ai_msg = await self._router.acompletion(
            messages=messages, model=self.model_sql
        )
        sql = parse_code(ai_msg.choices[0].message.content)
        return sql

    async def execute_sql(self, sql: str) -> tuple[str, bool, pd.DataFrame]:
        to_fix = True
        cnt = 0
        is_success = False
        data = pd.DataFrame()
        while to_fix:
            try:
                data = await self._client.execute_query(sql)
                to_fix = False
                is_success = True
            except Exception as e:
                logger.error(f"Error executing query: {str(e)}")
                traceback.print_exc()
                error = traceback.format_exception(e)[-1]
                if cnt > 2:
                    to_fix = False
                    data = pd.DataFrame()
                else:
                    sql = await self._fix_sql(sql, error)
                    cnt = cnt + 1
                    data = pd.DataFrame()
        return sql, is_success, data

    async def _extract_with_openai(
        self,
        messages: list[dict],
        model_type: str,
        reasoning_effort: Literal["low", "medium", "high"] = "medium",
    ) -> list[str]:
        """
        Extract tables using OpenAI models

        Args:
            messages (List[Dict]): Formatted messages
            model_type (str): Type of OpenAI model to use
            reasoning_effort (str): Reasoning effort level (default: "medium")

        Returns:
            list[str]: Extracted tables
        """

        if model_type == "o1":
            kwargs = {"reasoning_effort": "low"}
        elif model_type == "o1-mini":
            kwargs = {}
        else:
            kwargs = {"reasoning_effort": reasoning_effort}

        response = await self._router.acompletion(
            model=model_type, messages=messages, response_format=TablesListLLM, **kwargs
        )

        result = TablesListLLM(**parse_json(response.choices[0].message.content))
        return result.tables

    async def _extract_tables_without_reasoning(
        self, messages: list[dict], n: int
    ) -> list[str]:
        """
        Extract tables without using reasoning capabilities.

        Args:
            messages (list[dict]): Formatted messages for LLM
            n (int): Number of samples to generate

        Returns:
            list[str]: List of extracted table names
        """

        response = await self._router.acompletion(
            messages=messages,
            model=self.default_model,
            response_format=TablesListLLM,
            n=n,
        )

        all_tables = []
        for choice in response.choices:
            result = TablesListLLM(**parse_json(choice.message.content))
            all_tables.extend(result.tables)

        tables, counts = np.unique(all_tables, return_counts=True)
        return tables[counts > calculate_threshold(n)].tolist()

    async def _get_tables(
        self,
        request: str,
        n: int = 1,
        with_reasoning: bool = False,
        reasoning_model: str = "o3-mini",
        reasoning_effort: Literal["low", "medium", "high"] = "medium",
    ) -> list[str]:
        """
        Extract tables from text using specified LLM model and parameters.

        Args:
            request (str): Text to extract tables from
            n (int): Number of samples to generate (default: 1)
            with_reasoning (bool): Whether to use reasoning capabilities (default: False)
            reasoning_model (str): Model to use for reasoning (default: "DeepSeekLlama")
            reasoning_effort (str): Level of reasoning effort (default: "low")

        Returns:
            list[str]: Dictionary containing extracted table names

        Example:
            >>> result = await get_tables("Extract tables from this text", n=2)
            >>> print(result)
            ['table1', 'table2']
        """

        base_messages = [
            {
                "role": "system",
                "content": "You are an experienced Text Processing Specialist",
            },
            {"role": "user", "content": request},
        ]

        messages = [{"role": "user", "content": m["content"]} for m in base_messages]

        try:
            tables = await self._extract_with_openai(
                messages, reasoning_model, reasoning_effort
            )
        except Exception as e:
            n = 1 if reasoning_model == "simple" else n
            return await self._extract_tables_without_reasoning(messages, n)

        return tables

    async def _adjust_table_list(
        self,
        question: str,
        domain_concepts: str,
        business_rules: list[str],
        tables_descriptions: str,
        with_reasoning: bool,
        reasoning_model: str,
        reasoning_effort: Literal["low", "medium", "high"],
    ) -> list[str]:
        if with_reasoning:
            logger.info(
                f"..Analyze documentation: reasoning {with_reasoning}, model:{reasoning_model}"
            )
        else:
            logger.info(
                f"..Analyze documentation: reasoning {with_reasoning}, model: o1"
            )

        if with_reasoning:
            tbl_request = self.get_prompt("GET_TABLES_FROM_QUESTION_WITH_REASONING")
        else:
            tbl_request = self.get_prompt("GET_TABLES_FROM_QUESTION")

        request = tbl_request.substitute(
            {
                "question": question,
                "domain_concepts": domain_concepts,
                "business_rules": business_rules,
                "tables_descriptions": tables_descriptions,
            }
        )

        tables = await self._get_tables(
            request,
            n=7,
            with_reasoning=with_reasoning,
            reasoning_model=reasoning_model,
            reasoning_effort=reasoning_effort,
        )

        use_tables = np.union1d(tables, [])
        logger.info(f"Using table list: {use_tables}")

        return use_tables.tolist()

    async def get_related_tables(
        self,
        norm_question: str,
        main_clause: str,
        concepts: str,
        list_tables_max: list[str],
        seek_table_reasoning: bool = False,
        reasoning_model: str = "o3-mini",
        reasoning_effort: Literal["low", "medium", "high"] = "medium",
    ) -> tuple[str, list[Any]]:
        """
        Retrieves tables related to a normalized question using various querying strategies.

        Args:
            norm_question (str): Normalized version of the user's question
            main_clause (str): Main clause extracted from the question
            concepts (str): Concepts identified in the question
            list_tables_max (List[str]): Maximum list of tables to consider
            seek_table_reasoning (bool): Whether to use advanced reasoning
            reasoning_model (str): Model to use for reasoning
            reasoning_effort (Literal["low", "medium", "high"]): Level of reasoning effort
            **kwargs: Additional keyword arguments

        Returns:
            tuple[str, list[Any]]: A tuple containing table documentation and table names
        """

        def get_docs(x):
            res = json.loads(x)
            return (
                [res["table"]]
                + json.loads(res["dependencies"])
                + json.loads(res["connected_tables"])
            )

        domain_instr = await self._vector_store.get_domain_instructions()

        # categories = ["description", "description_dependencies", "table_name", "connected_tables", "entity"]
        query_params = [
            {
                "query_texts": [main_clause],
                "where": {"$or": [{"category": "description"}, {"category": "entity"}]},
                "n_results": 10,
            },
            {
                "query_texts": [norm_question],
                "where": {"$or": [{"category": "description"}, {"category": "entity"}]},
                "n_results": 10,
            },
            {
                "query_texts": [concepts],
                "where": {
                    "$or": [{"category": "connected_tables"}, {"category": "entity"}]
                },
                "n_results": 20,
            },
            {
                "query_texts": [main_clause],
                "where": {"category": "table_name"},
                "n_results": 10,
            },
            {
                "query_texts": [norm_question],
                "where": {"category": "table_name"},
                "n_results": 10,
            },
            {
                "query_texts": [concepts],
                "where": {"category": "table_name"},
                "n_results": 10,
            },
        ]

        tables_query = []
        for param in query_params:
            ruery_res = await self._vector_store.query_documentation(**param)
            ruery_res = ruery_res["metadatas"][0]
            for r in ruery_res:
                tables_query.append(
                    json.dumps(
                        {
                            "table": r["table"],
                            "dependencies": r["dependencies"],
                            "connected_tables": r["connected_tables"],
                        }
                    )
                )

        tables_query = np.unique(tables_query)
        tables_query = list(map(lambda x: get_docs(x), tables_query))

        raw_tables = []
        for x in tables_query:
            raw_tables = raw_tables + x

        tbls = await self.get_tables_from_business_rules(question=norm_question, model=reasoning_model, reasoning_effort=reasoning_effort)
        raw_tables = np.unique(raw_tables + tbls)
        tbl_descr, _tables = await self.return_table_docs(raw_tables)

        use_tables = await self._adjust_table_list(
            question=norm_question,
            domain_concepts=domain_instr,
            business_rules=self._business_rules,
            tables_descriptions=tbl_descr,
            with_reasoning=seek_table_reasoning,
            reasoning_model=reasoning_model,
            reasoning_effort=reasoning_effort,
        )

        docs_to_sql, _tables = await self.return_table_docs(
            np.unique(use_tables + list_tables_max).tolist()
        )
        return docs_to_sql, _tables

    @abstractmethod
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
        pass

    async def train(
        self,
        question: str = None,
        sql: str = None,
        ddl: str = None,
        documentation: str = None,
        plan: TrainingPlan = None,
        metadatas=None,
        add_name=None,
    ) -> str:
        if documentation:
            logger.info("Adding documentation....")
            return await self.add_documentation(
                documentation, question=question, metadatas=metadatas, add_name=add_name
            )

        if sql:
            if question is None:
                question = await self.generate_question(sql)
                logger.info(f"Question generated with sql: {question} \nAdding SQL...")
            return await self.add_question_sql(
                question=question, sql=sql, metadatas=metadatas
            )

        if ddl:
            logger.info(f"Adding ddl: {ddl}")
            return await self.add_ddl(ddl)

        if plan:
            for item in plan._plan:
                if item.item_type == TrainingPlanItem.ITEM_TYPE_DDL:
                    await self.add_ddl(item.item_value)
                elif item.item_type == TrainingPlanItem.ITEM_TYPE_IS:
                    await self.add_documentation(item.item_value)
                elif item.item_type == TrainingPlanItem.ITEM_TYPE_SQL:
                    await self.add_question_sql(
                        question=item.item_name, sql=item.item_value
                    )

    async def extract_sql_tables_json(self, selected_tables: str) -> list[dict]:
        response = await self._router.acompletion(
            model=self.default_model,
            messages=[
                {
                    "role": "user",
                    "content": f"""Extract a list of tables and the description of why these tables are needed as JSON from the text: 
{selected_tables}.

You should return the result in JSON format:
{{
    "tables": [
        {{
            "table": "fundraisings",
            "why": "To get the details of fundraising actions"
        }}
    ]
}}
""",
                }
            ],
            response_format=TableWhyListLLM,
        )
        tables = TableWhyListLLM(**parse_json(response.choices[0].message.content))
        return [table.model_dump() for table in tables.tables]

    @staticmethod
    def get_training_plan_generic(df: pd.DataFrame) -> TrainingPlan:
        """
        This method is used to generate a training plan from an information schema dataframe.

        Basically what it does is breaks up INFORMATION_SCHEMA.COLUMNS into groups of table/column descriptions that can be used to pass to the LLM.

        Args:
            df (pd.DataFrame): The dataframe to generate the training plan from.

        Returns:
            TrainingPlan: The training plan.
        """
        # For each of the following, we look at the df columns to see if there's a match:
        database_column = df.columns[
            df.columns.str.lower().str.contains("database")
            | df.columns.str.lower().str.contains("table_catalog")
        ].to_list()[0]
        schema_column = df.columns[
            df.columns.str.lower().str.contains("table_schema")
        ].to_list()[0]
        table_column = df.columns[
            df.columns.str.lower().str.contains("table_name")
        ].to_list()[0]
        columns = [database_column, schema_column, table_column]
        candidates = ["column_name", "data_type", "comment"]
        matches = df.columns.str.lower().str.contains("|".join(candidates), regex=True)
        columns += df.columns[matches].to_list()

        plan = TrainingPlan([])

        for database in df[database_column].unique().tolist():
            for schema in (
                df.query(f'{database_column} == "{database}"')[schema_column]
                .unique()
                .tolist()
            ):
                for table in (
                    df.query(
                        f'{database_column} == "{database}" and {schema_column} == "{schema}"'
                    )[table_column]
                    .unique()
                    .tolist()
                ):
                    df_columns_filtered_to_table = df.query(
                        f'{database_column} == "{database}" and {schema_column} == "{schema}" and {table_column} == "{table}"'
                    )
                    doc = f"The following columns are in the {table} table in the {database} database:\n\n"
                    doc += df_columns_filtered_to_table[columns].to_markdown()

                    plan._plan.append(
                        TrainingPlanItem(
                            item_type=TrainingPlanItem.ITEM_TYPE_IS,
                            item_group=f"{database}.{schema}",
                            item_name=table,
                            item_value=doc,
                        )
                    )

        return plan

    async def generate_question(self, sql: str) -> str:
        response = await self._router.acompletion(
            model=self.default_model,
            temperature=0.7,
            messages=[
                {
                    "role": "system",
                    "content": "The user will give you SQL and you will try to guess what the business question this query is answering. "
                    "Return just the question without any additional explanation. Do not reference the table name in the question.",
                },
                {"role": "user", "content": sql},
            ],
        )

        return response.choices[0].message.content

    async def expand_question(self, question: str, code: str) -> list[str]:
        request = self.get_prompt("EXPAND_QUESTION").substitute(
            {"question": question, "code": code}
        )

        messages = [
            {
                "role": "system",
                "content": "You are an experienced Text Processing Specialist",
            },
            {"role": "user", "content": request},
        ]

        try:
            ai_msg = await self._router.acompletion(
                messages=messages,
                model=self.default_model,
                response_format=ExpandQuestionLLM,
            )
            questions = ExpandQuestionLLM(
                **parse_json(ai_msg.choices[0].message.content)
            ).questions
        except Exception as e:
            logger.error(f"Error expanding question: {str(e)}")
            traceback.print_exc()
            questions = []

        return questions

    async def get_domain_specific_mapping(self, summary: str) -> str:
        messages = [
            {"role": "system", "content": summary},
            {"role": "user", "content": self.get_prompt_string("DOMAIN_EXTRACT")},
        ]

        ai_msg = await self._router.acompletion(
            messages=messages, model=self.default_model
        )
        answer = ai_msg.choices[0].message.content
        return answer

    async def add_question_sql(self, question: str, sql: str, **kwargs) -> str:
        return await self._vector_store.add_question_sql(question, sql, **kwargs)

    async def add_ddl(self, ddl: str, **kwargs) -> str:
        return await self._vector_store.add_ddl(ddl, **kwargs)

    async def add_documentation(self, documentation: str, **kwargs) -> str:
        return await self._vector_store.add_documentation(documentation, **kwargs)

    async def get_all_documentation(self) -> list:
        return await self._vector_store.get_all_documentation()

    async def return_table_docs(self, list_tables: list[str]) -> tuple:
        return await self._vector_store.return_table_docs(list_tables)

    async def get_document_by_name(self, name: str) -> dict:
        return await self._vector_store.get_document_by_name(name)

    async def get_examples_by_question_name(self, name: str) -> dict:
        return await self._vector_store.get_examples_by_question_name(name)

    async def remove_training_data(self, ids: list[str]) -> None:
        await self._vector_store.remove_training_data(ids)

    async def add_question_tables_relation(self, question: str, tables: str) -> str:
        question_doc_json = json.dumps({"question": question, "tables": tables})
        id = deterministic_uuid(question_doc_json) + "-qst-doc"
        await self._vector_store.add_question_tables_relation(
            id, question_doc_json, question
        )
        return id

    @abstractmethod
    async def learn_json_document(
        self, document_data: dict, allow_replace: bool = False
    ):
        pass

    @abstractmethod
    async def learn_sql(self, question: str, sql_code: str) -> None:
        pass
