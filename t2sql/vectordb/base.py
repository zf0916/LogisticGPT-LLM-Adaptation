from abc import ABC, abstractmethod
import pandas as pd
from typing import Any


class VectorStore(ABC):
    def __init__(self, config: dict):
        self._config = config

    @abstractmethod
    async def generate_embedding(self, data: str, **kwargs) -> list[float]:
        pass

    @abstractmethod
    async def add_documentation(
        self,
        documentation: str,
        question: str = None,
        metadatas=None,
        add_name=None,
        **kwargs,
    ) -> str:
        pass

    @abstractmethod
    async def add_question_sql(
        self, question: str, sql: str, metadatas=None, **kwargs
    ) -> str:
        pass

    @abstractmethod
    async def add_ddl(self, ddl: str, **kwargs) -> str:
        pass

    @abstractmethod
    async def add_question_tables_relation(
        self, id: str, question_doc_json: str, question: str
    ):
        pass

    @abstractmethod
    async def get_all_documentation(self) -> list:
        pass

    @abstractmethod
    async def get_document_by_name(self, document_name: str) -> dict:
        pass

    @abstractmethod
    async def get_examples_by_question_name(self, question: str) -> dict:
        pass

    @abstractmethod
    async def delete_documents_by_ids(self, ids: list[str]):
        pass

    @staticmethod
    @abstractmethod
    def extract_documents(query_results: Any) -> list:
        pass

    @abstractmethod
    def remove_collection(self, collection_name: str) -> bool:
        return False

    @abstractmethod
    async def remove_training_data(self, id: list[str]) -> bool:
        return False

    @abstractmethod
    def get_training_data(self, **kwargs) -> pd.DataFrame:
        pass

    @staticmethod
    @abstractmethod
    def filter_by_distance_score_sql(
        result: list,
        query_result: Any,
        min_score: float,
        best_score: float,
        break_if_close: bool = False,
        indexes: list = None,
    ) -> tuple[list, bool, list]:
        """
        Filter query results based on distance scores and extract table information.

        Args:
            result: List of query results
            query_result: QueryResult object containing distances and metadata
            min_score: Maximum distance threshold for filtering
            best_score: Threshold for determining best examples
            break_if_close: Whether to stop processing after finding a best example
            indexes: List of indexes to consider. If None, uses all indexes.

        Returns:
            tuple containing:
            - filtered_results: List of filtered results with structure metadata
            - has_best_example: Boolean indicating if a best example was found
            - unique_tables: List of unique table names extracted from structures
        """
        pass

    @abstractmethod
    async def get_related_sql(self, question: str) -> Any:
        pass

    @abstractmethod
    def get_related_ddl(self, question: str, **kwargs) -> list:
        pass

    @abstractmethod
    def get_related_documentation(self, question: str, **kwargs) -> list:
        pass

    @abstractmethod
    async def query_documentation(self, **kwargs) -> Any:
        pass

    @abstractmethod
    async def return_table_docs(self, list_tables: list[str]):
        """
        Get documentation and metadata for a list of tables.

        Args:
            list_tables (list): List of table names

        Returns:
            tuple: (documentation string, list of table names)
        """
        pass

    @abstractmethod
    async def get_domain_instructions(self) -> str:
        """
        Retrieves domain-specific instructions from the questions tables collection.

        Returns:
            str: Domain-specific instructions extracted from the collection. Returns an empty
                 string if no matching document is found.
        """
        pass
