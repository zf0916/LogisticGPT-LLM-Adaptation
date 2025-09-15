import json
import numpy as np
from t2sql.base import BaseText2SQLAgent
from t2sql.vectordb.chromadb import VectorStore
from t2sql.utils import parse_json, parse_code, load_examples
from abc import ABC
from pydantic import BaseModel
import os
from t2sql.utils import get_config, logger, DEFAULT_DESCRIPTOR_FILE_NAME, deterministic_uuid
from pathlib import Path


class TablesListLLM(BaseModel):
    tables: list[str]


class Column(BaseModel):
    column: str
    description: str


class ProcessDocumentLLM(BaseModel):
    name: str
    summary: str
    purpose: str
    dependencies_thoughts: str
    keys: list[str]
    connected_tables: list[str]
    columns: list[Column]
    strong_entities: list[str] | None = None
    entities: list[str] | None = None
    document_name: str | None = None

    @property
    def columns_normalized(self) -> str:
        columns = ""
        for col in self.columns:
            columns = columns + f"`{col.column}`: {col.description}\n"
        return columns


class EntitiesLLM(BaseModel):
    entities: list[str]


class TextIngestion(BaseText2SQLAgent, ABC):
    def __init__(self, config: dict, vector_store: VectorStore):
        super().__init__(config, vector_store)
        self._docs_md_folder = config.get("docs_md_folder")
        self._docs_json_folder = config.get("docs_json_folder")
        self._examples_folder = config.get("examples_folder")
        self._examples_extended_folder = config.get("examples_extended_folder")
        self._docs_ddl_folder = config.get("docs_ddl_folder")

    async def _process_text_document(self, document: str) -> ProcessDocumentLLM:
        content = self.get_prompt("PROCESS_DOCUMENT").substitute({"document": document})
        messages = [
            {"role": "system", "content": "You are an experienced Data Engineer"},
            {"role": "user", "content": content},
        ]
        completion_result = await self._router.acompletion(
            self.default_model,
            messages=messages,
            response_format=ProcessDocumentLLM,
        )

        result = ProcessDocumentLLM(
            **parse_json(completion_result.choices[0].message.content)
        )
        return result

    async def _extract_entities(
        self, processed_document: ProcessDocumentLLM
    ) -> ProcessDocumentLLM:
        request = f"Table: {processed_document.name}\n{processed_document.summary}. {processed_document.purpose}"
        messages = [
            {
                "role": "user",
                "content": self.get_prompt("ENTITIES_DOCUMENT").substitute(
                    {"table": request}
                ),
            }
        ]

        result = await self._router.acompletion(
            model=self.default_model,
            messages=messages,
            response_format=EntitiesLLM,
            n=7,
            temperature=1,
        )

        entities = []
        for r in result.choices:
            entities.extend(EntitiesLLM(**parse_json(r.message.content)).entities)

        entities, counts = np.unique(entities, return_counts=True)
        processed_document.strong_entities = entities[counts > 3].tolist()
        processed_document.entities = entities[counts > 2].tolist()

        return processed_document

    async def _extract_tables(self, text: str) -> list[str]:
        messages = [
            {
                "role": "user",
                "content": self.get_prompt("EXTRACT_TABLES_TEXT").substitute(
                    {"text": text}
                ),
            }
        ]
        try:
            ai_msg = await self._router.acompletion(
                model=self.default_model,
                messages=messages,
                response_format=TablesListLLM,
            )
            answer = TablesListLLM(**parse_json(ai_msg.choices[0].message.content))
            return answer.tables
        except Exception as e:
            logger.error(f"Cannot extract tables from document. Error: {e}")
            return []

    async def _generate_embeddings(
        self, normalized_document: ProcessDocumentLLM
    ) -> None:
        description = self.get_prompt("DESCRIPTION").substitute(
            normalized_document.model_dump()
        )
        description_llm = self.get_prompt("DESCRIPTION_LLM").substitute(
            normalized_document.model_dump()
        )
        tables = await self._extract_tables(normalized_document.dependencies_thoughts)
        tables = json.dumps(tables)

        description_full = (
            f"{description_llm}\nCOLUMNS:\n{normalized_document.columns_normalized}"
        )
        await self.train(
            documentation=description_full,
            question=description,
            metadatas={
                "table": normalized_document.name,
                "connected_tables": json.dumps(normalized_document.connected_tables),
                "keys": json.dumps(normalized_document.keys),
                "category": "description",
                "dependencies": tables,
                "document_name": normalized_document.document_name,
            },
        )

        await self.train(
            documentation=description_llm,
            question=None,
            metadatas={
                "table": normalized_document.name,
                "connected_tables": json.dumps(normalized_document.connected_tables),
                "keys": json.dumps(normalized_document.keys),
                "category": "description_dependencies",
                "dependencies": tables,
                "document_name": normalized_document.document_name,
            },
        )

        await self.train(
            documentation=normalized_document.name,
            question=None,
            metadatas={
                "table": normalized_document.name,
                "description": description_llm,
                "description_full": description_full,
                "dependencies": tables,
                "connected_tables": json.dumps(normalized_document.connected_tables),
                "keys": json.dumps(normalized_document.keys),
                "category": "table_name",
                "document_name": normalized_document.document_name,
            },
        )

        await self.train(
            documentation=json.dumps(normalized_document.connected_tables),
            question=None,
            metadatas={
                "table": normalized_document.name,
                "dependencies": tables,
                "connected_tables": json.dumps(normalized_document.connected_tables),
                "keys": json.dumps(normalized_document.keys),
                "category": "connected_tables",
                "document_name": normalized_document.document_name,
            },
            add_name=normalized_document.name,
        )

        for ent in normalized_document.entities:
            await self.train(
                documentation=f"{ent}",
                question=None,
                metadatas={
                    "table": normalized_document.name,
                    "dependencies": tables,
                    "connected_tables": json.dumps(
                        normalized_document.connected_tables
                    ),
                    "keys": json.dumps(normalized_document.keys),
                    "category": "entity",
                    "document_name": normalized_document.document_name,
                },
                add_name=normalized_document.name,
            )

    async def learn_json_document(
        self, document_data: ProcessDocumentLLM, allow_replace: bool = False
    ) -> None:
        """
        Processes and trains on a single documentation JSON file.

        Args:
            document_data (dict): JSON content containing table documentation and metadata
            allow_replace (bool): if allow replacing the document with the same name

        Process:
            1. Extracts tables mentioned in the documentation
            2. Generates descriptions using templates
            3. Processes column information
            4. Creates multiple training entries for different aspects:
               - Full documentation with table descriptions
               - Dependencies and relationships
               - Table names and metadata
               - Connected tables
               - Entity relationships

        Notes:
            - Uses LLM for table extraction and processing
            - Creates multiple training entries with different categories
            - Handles complex relationships between tables
            - Stores metadata including dependencies and keys
        """

        existing_documents = await self._vector_store.get_document_by_name(
            document_data.document_name
        )
        existing_documents = existing_documents["ids"]

        if len(existing_documents) > 0 and not allow_replace:
            raise Exception(
                f"Documents with such name '{document_data.document_name}' exists"
            )

        logger.debug(f"Replacing documents {existing_documents}")
        await self._vector_store.remove_training_data(existing_documents)

        try:
            await self._generate_embeddings(document_data)
        except Exception as e:
            logger.error(f"Cannot generate embeddings for the document, error: {e}")
            raise Exception("Cannot generate embeddings for the document")

    async def learn_md_document(
        self, md_name: str, md_content: str
    ) -> ProcessDocumentLLM:
        """
        Processes a single markdown document to extract structured information.

        Args:
            md_name (str): Name of the markdown document
            md_content (str): Content of the markdown document to process

        Returns:
            dict: Structured description including:
                - name: Table name
                - summary: Table summary
                - purpose: Table purpose
                - entities: Extracted entities
                - strong_entities: Frequently mentioned entities

        Process:
            1. Processes document using GPT-4 for initial parsing
            2. Extracts entities and relationships
            3. Identifies strong entities based on frequency
            4. Combines information into structured format

        Notes:
            - Uses advanced LLM processing
            - Performs multiple passes for entity extraction
            - Differentiates between regular and strong entities
            - Returns comprehensive structured description
        """
        with open(f"{self._docs_md_folder}/{md_name}", "w") as file:
            file.write(md_content)

        processed_doc = await self._process_text_document(md_content)
        result = await self._extract_entities(processed_doc)
        result.document_name = md_name

        return result

    async def learn_sql(self, question: str, sql_code: str) -> None:
        """
        Adds a single SQL example to the training data with various transformations and expansions.

        Args:
            question (str): The natural language question to train on
            sql_code (str): The corresponding SQL query

        Process:
            1. Normalizes and structures the original question
            2. Trains on the original question
            3. Extracts and trains on the concepts/entities
            4. Trains on the normalized version of the question
            5. Generates additional similar questions and trains on them

        Notes:
            - Uses text_analyzer for question normalization and expansion
            - Stores structured metadata with each training example
            - Handles errors during additional question generation gracefully
            - Prints progress information during the training process
        """
        result = await self.normalize_and_structure(question=question, sql=sql_code)
        logger.debug(f"1. Train on question: {question}")
        result_json = result.model_dump_json()
        await self.train(
            question,
            sql_code,
            metadatas={"structure": result_json, "question": question},
        )
        logger.debug(f"2. Train on concepts: {result.requested_entities}")
        await self.train(
            result.requested_entities,
            sql_code,
            metadatas={"structure": result_json, "question": question},
        )
        logger.debug(f"3. Train on normalized question: {result.normalized_question}")
        await self.train(
            result.normalized_question,
            sql_code,
            metadatas={"structure": result_json, "question": question},
        )
        logger.debug("4. Generate additional questions...")
        questions = await self.expand_question(result.normalized_question, sql_code)
        k = 5
        for qw in questions:
            try:
                logger.debug(f"{k}. Train on question: {qw}")
                k += 1
                await self.train(
                    qw,
                    sql_code,
                    metadatas={"structure": result_json, "question": question},
                )
            except Exception as e:
                logger.error(f"Error training on data for '{qw}': {str(e)}")

    async def train_on_documentation_json(self) -> None:
        """
        Trains the system on JSON-formatted documentation files.

        Notes:
            - Processes each JSON file in the specified folder
            - Uses learn_document for individual document processing
            - Displays training progress as percentage
            - Expects JSON files with specific documentation structure
        """
        if self._docs_json_folder is None:
            raise ValueError("Specify 'docs_json_folder' to train the data.")

        progress = 0.0
        files = os.listdir(self._docs_json_folder)
        files_count = len(files)

        for filename in files:
            input_path = os.path.join(self._docs_json_folder, filename)
            with open(input_path, "r") as file:
                json_content = json.load(file)
                await self.learn_json_document(json_content, allow_replace=True)
                progress += 1.0
                logger.info(
                    f"Training on documentation from ({self._docs_json_folder}) [{progress}/{files_count}] | {progress / files_count * 100}"
                )

    async def train_on_examples(self) -> None:
        """
        Processes and trains on multiple SQL examples from a directory of JSON files.

        Notes:
            - Processes JSON files with either 'sql_1' or 'sql' keys
            - Uses add_single_sql_v1 for processing each example
            - Tracks and displays training progress as percentage
            - Each JSON file should contain a question and corresponding SQL query
            - Handles multiple SQL formats for backward compatibility
        """
        if self._examples_extended_folder is None:
            raise ValueError("Specify 'examples_extended_folder' to train the data.")

        progress = 0.0
        files = os.listdir(self._examples_extended_folder)
        files_count = len(files)

        for filename in files:
            input_path = os.path.join(self._examples_extended_folder, filename)
            with open(input_path, "r") as file:
                state = json.load(file)
                sql_code = state.get("sql_1") or state.get("sql")
                await self.learn_sql(state["question"], sql_code)
                progress += 2.0
                logger.info(
                    f"Training on example [{progress}/{files_count}] | {progress / files_count * 100}"
                )

    async def train_on_ddl(self) -> None:
        """
        Trains the system on DDL (Data Definition Language) documentation.

        Notes:
            - Processes each DDL file in the specified folder
            - Adds DDL information to the training system
            - Tracks and displays training progress
            - Enhances system's understanding of table structures
        """
        if self._docs_ddl_folder is None:
            raise ValueError("Specify 'docs_ddl_folder' to train the data.")

        progress = 0.0
        files = os.listdir(self._docs_ddl_folder)
        files_count = len(files)

        for filename in files:
            input_path = os.path.join(self._docs_ddl_folder, filename)
            with open(input_path, "r") as file:
                ddl_content = file.read()
                await self.train(ddl=ddl_content)
                progress += 1.0
                logger.info(
                    f"Training on ddl [{progress}/{files_count}] | {progress / files_count * 100}"
                )

    async def train_on_tables_question_relation(self) -> None:
        """
        Trains the system on relationships between questions and tables.

        Process:
            - Reads JSON files from training data directory
            - Processes question-table relationships
            - Adds relationships to the training system
            - Tracks and displays progress

        Notes:
            - Expects JSON files with either 'tables_json' or 'tables' format
            - Maintains relationships between questions and their relevant tables
            - Used for improving table selection in queries
        """
        if self._examples_extended_folder is None:
            raise ValueError("Specify 'examples_extended_folder' to train the data.")

        progress = 0.0
        files = os.listdir(self._examples_extended_folder)
        files_count = len(files)

        for filename in files:
            input_path = os.path.join(self._examples_extended_folder, filename)
            with open(input_path, "r") as file:
                state = json.load(file)
                tables = state.get("tables_json") or state.get("tables")
                await self.add_question_tables_relation(state["question"], tables)
                progress += 2.0
                logger.info(
                    f"Training on tables relation to questions [{progress}/{files_count}] | {progress / files_count * 100}"
                )

    async def train_on_information_schema(
        self, train_if_doc_exists: bool = False
    ) -> None:
        # TODO check for other DBs column names and move to t2sql/sql/client.py the logic of extraction
        # make it use async and speed up
        if self._database and self._schema:
            query = f"SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE table_catalog = '{self._database}' AND table_schema = '{self._schema}'"
        elif self._database and self._schema is None:
            query = f"SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE table_catalog = '{self._database}'"
        else:
            query = f"SELECT * FROM INFORMATION_SCHEMA.COLUMNS"

        df_information_schema = await self._client.execute_query(query)

        inf = df_information_schema[
            ["table_catalog", "table_schema", "table_name", "column_name", "data_type"]
        ]
        list_tables = list(
            map(
                lambda x: f"{x[0]}.{x[1]}.{x[2]}",
                inf.groupby(["table_catalog", "table_schema", "table_name"]).count().index.to_list(),
            )
        )

        tables_count = len(inf.groupby(["table_catalog", "table_schema", "table_name"]))
        progress = 0
        for tbl in inf.groupby(["table_catalog", "table_schema", "table_name"]):
            if self._database and self._schema:
                fname = Path(f"{self._docs_md_folder}/{tbl[0][2]}.md")
            else:
                fname = Path(f"{self._docs_md_folder}/{tbl[0][0]}_{tbl[0][1]}_{tbl[0][2]}.md")

            if fname.is_file() and not train_if_doc_exists:
                logger.debug(f"Do not replace the document {fname}")
                continue

            tbl_col = json.dumps(
                tbl[1][["column_name", "data_type"]].to_dict("records")
            )
            table_name = tbl[0][2]

            if self._database and self._schema:
                table_info = f"Table: {table_name}\n\nTotal list of Tables: {list_tables}\n\nTable Info: {tbl_col}"
            else:
                database_name = tbl[0][0]
                schema_name = tbl[0][1]
                table_info = f"Table: {database_name}.{schema_name}.{table_name}\n\nTotal list of Tables: {list_tables}\n\nTable Info: {tbl_col}"

            content = self.get_prompt("PREPARE_MD_FROM_SCHEMA").substitute(
                {"table": table_info}
            )
            messages = [
                {"role": "user", "content": content},
            ]
            completion_result = await self._router.acompletion(
                self.default_model,
                messages=messages,
            )
            md_content = completion_result.choices[0].message.content

            with open(fname, "w") as file:
                file.write(md_content)

            json_content = await self.learn_md_document(fname.name, md_content)

            await self.learn_json_document(json_content, allow_replace=True)

            progress += 1.0
            logger.info(
                f"Training on documentation for {table_name} [{progress}/{tables_count}] | {progress / tables_count * 100}"
            )

    async def train_in_domain_specific(self):
        """
        Trains the system on domain-specific mappings and relationships.

        Returns:
            str: The extracted domain-specific mapping result

        Process:
            1. Reads examples from the configured folder
            2. Builds context from questions and table relationships
            3. Extracts domain-specific mappings
            4. Stores the mappings for future use

        Notes:
            - Creates a comprehensive context of question-table relationships
            - Processes both 'tables_json' and 'tables' formats
            - Adds explanatory context for table usage
            - Used for improving domain-specific understanding
        """
        if self._examples_extended_folder is None:
            raise ValueError("Specify 'examples_extended_folder' to train the data.")

        files = os.listdir(self._examples_extended_folder)
        context = ""

        for filename in files:
            input_path = os.path.join(self._examples_extended_folder, filename)
            with open(input_path, "r") as file:
                state = json.load(file)
                context = f'''{context}Question: "{state["question"]}"\nTables:'''
                tables = state.get("tables_json") or state.get("tables")
                for tbl in tables:
                    context = (
                        f"""{context}\n{tbl["table"]}. Explanation: {tbl["why"]}"""
                    )
                context = f"""{context}\n"""

        result = await self.get_domain_specific_mapping(context)
        await self.add_question_tables_relation(
            "EXTRACTED DOMAIN-SPECIFIC MAPPING", result
        )

    async def expand_example_structure(self, example: dict) -> dict:
        if self._examples_extended_folder is None:
            raise ValueError("Specify 'examples_extended_folder' to extend the data.")

        tables = self.get_information_about_all_tables()
        rules = json.dumps(self._business_rules)

        if not os.path.exists(self._examples_extended_folder):
            os.makedirs(self._examples_extended_folder)

        messages = [
            {"role": "system", "content": "You're SQL expert."},
            {
                "role": "user",
                "content": "Here is the information about tables in the database: \n"
                + tables
                + "\n"
                + "You should select the tables needed to answer user question: "
                + example["question"]
                + "\n"
                + "Here are the rules you should follow when answering question: "
                + rules
                + "\n"
                + "The SQL query that answers the question is: "
                + example["sql"]
                + "\n"
                + "You should return the list of tables needed and why they are needed."
                + "You should return list of needed tables as text",
            },
        ]
        response = await self._router.acompletion(
            messages=messages, model=self.default_model
        )
        example["selection of talbes"] = response.choices[0].message.content
        example["tables"] = await self.extract_sql_tables_json(response)
        uuid = deterministic_uuid(example['question'])
        with open(
            f"{self._examples_extended_folder}/{uuid}.json",
            "w",
        ) as file:
            json.dump(example, file)
        return example

    async def remove_example(self, question: str) -> None:
        if self._examples_extended_folder is None:
            raise ValueError("Specify 'examples_extended_folder' to extend the data.")

        try:
            examples = await self.get_examples_by_question_name(question)
            await self.remove_training_data(examples["ids"])
            uuid = deterministic_uuid(question)
            os.remove(
                f"{self._examples_extended_folder}/{uuid}.json"
            )
        except Exception as e:
            logger.warning(str(e))

    async def remove_document(self, md_name: str) -> None:
        if self._docs_md_folder is None:
            raise ValueError("Specify 'docs_md_folder' to extend the data.")

        try:
            docs = await self.get_document_by_name(md_name)
            await self.remove_training_data(docs["ids"])
            os.remove(f"{self._docs_md_folder}/{md_name}")
        except Exception as e:
            logger.warning(str(e))

    async def expand_examples_structure(self) -> None:
        if self._examples_folder is None or self._examples_extended_folder is None:
            raise ValueError(
                "Specify 'examples_folder' and 'examples_extended_folder' to extend the data."
            )

        if not os.path.exists(self._examples_extended_folder):
            os.makedirs(self._examples_extended_folder)

        input_path = os.path.join(self._examples_folder, "examples.json")
        with open(input_path, "r") as file:
            data = json.load(file)
            for idx, example in enumerate(data):
                example_description = await self.expand_example_structure(example)
                with open(f"{self._examples_extended_folder}/{idx}.json", "w") as fo:
                    json.dump(example_description, fo, indent=4)
                logger.info(f"Done {idx}/{len(data)}")

    def get_information_about_all_tables(self) -> str:
        if self._docs_md_folder is None:
            raise ValueError("Specify 'docs_md_folder' to process the data.")

        files = os.listdir(self._docs_md_folder)
        tables = ""

        for filename in files:
            input_path = os.path.join(self._docs_md_folder, filename)

            with open(input_path, "r") as file:
                md_content = file.read()
                tables = tables + md_content + "\n"
        return tables

    async def generate_json_from_md_documentation(self) -> None:
        """
        Processes all markdown documentation files into structured JSON format.

        Process:
            1. Reads markdown files from input directory
            2. Processes each file using process_single_doc
            3. Saves structured output as JSON
            4. Tracks and displays processing progress

        Notes:
            - Converts markdown documentation to structured format
            - Maintains file naming convention (.md -> .json)
            - Creates structured, machine-readable documentation
            - Preserves all extracted metadata and relationships
        """
        if self._docs_md_folder is None or self._docs_json_folder is None:
            raise ValueError(
                "Specify 'docs_md_folder' and 'docs_json_folder' to process the data."
            )

        if not os.path.exists(self._docs_json_folder):
            os.makedirs(self._docs_json_folder)

        progress = 0.0
        files = os.listdir(self._docs_md_folder)
        files_count = len(files)

        for filename in files:
            input_path = os.path.join(self._docs_md_folder, filename)
            with open(input_path, "r") as file:
                md_content = file.read()
                descr = await self.learn_md_document(filename, md_content)
                fn = filename.replace(".md", ".json")
                with open(os.path.join(self._docs_json_folder, fn), "w") as file:
                    json.dump(descr.model_dump(), file)
                progress += 1.0
                logger.info(
                    f"Training on documentation [{progress}/{files_count}] | {progress / files_count * 100}"
                )

    async def generate_ddl_from_md_documentation(self, md_content: str) -> str:
        messages = [
            {
                "role": "system",
                "content": f"You are an expert SQL query generator. You generate {self._dialect} DDL SQL Queries based on the provided text description of tables. You return only SQL query and nothing else.",
            },
            {
                "role": "user",
                "content": f"Generate DDL SQL for the table with the next description: {md_content}.",
            },
        ]
        response = await self._router.acompletion(
            model=self.default_model, messages=messages, temperature=0.7
        )

        code = parse_code(response.choices[0].message.content)
        return code

    async def generate_ddls_from_md_documentation(self) -> None:
        """
        Generates DDL SQL statements from markdown documentation files.

        Process:
            1. Creates output directory if it doesn't exist
            2. Reads markdown documentation files
            3. Generates Redshift DDL SQL queries from documentation
            4. Saves generated DDL to output files

        Notes:
            - Uses expert SQL generation system
            - Converts markdown descriptions to DDL
            - Maintains file naming convention (.md -> .sql)
            - Tracks and displays progress
        """
        if self._docs_md_folder is None or self._docs_ddl_folder is None:
            raise ValueError(
                "Specify 'docs_md_folder' and 'docs_ddl_folder' to process the data."
            )

        if not os.path.exists(self._docs_ddl_folder):
            os.makedirs(self._docs_ddl_folder)

        progress = 0.0
        files = os.listdir(self._docs_md_folder)
        files_count = len(files)

        for filename in files:
            input_path = os.path.join(self._docs_md_folder, filename)
            with open(input_path, "r") as file:
                md_content = file.read()
                code = await self.generate_ddl_from_md_documentation(md_content)
                output_path = os.path.join(
                    self._docs_ddl_folder, filename.replace("md", "sql")
                )
                with open(output_path, "w") as file:
                    file.write(code)
                progress += 1.0
                logger.info(
                    f"Training on documentation [{progress}/{files_count}] | {progress / files_count * 100}"
                )

    def update_business_rules(self, rules: list[str]) -> None:
        config = get_config(self._descriptor_folder)
        config["business_rules"] = rules
        with open(
                os.path.join(self._descriptor_folder, DEFAULT_DESCRIPTOR_FILE_NAME), "w"
        ) as f:
            f.write(json.dumps(config))
        self.refresh_business_rules(rules)

    def update_prompts(self, sql_instruction: str) -> None:
        config = get_config(self._descriptor_folder)
        config["prompts"] = {"DEFAULT_SQL_INSTRUCTIONS": sql_instruction}
        with open(
                os.path.join(self._descriptor_folder, DEFAULT_DESCRIPTOR_FILE_NAME), "w"
        ) as f:
            f.write(json.dumps(config))
        self.refresh_prompts()

    def load_examples(self) -> list[dict]:
        return load_examples(self._examples_extended_folder)
