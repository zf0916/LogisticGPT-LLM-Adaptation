import asyncio
from sqlalchemy import text
from typing import AsyncIterator
from abc import ABC, abstractmethod
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncConnection,
    create_async_engine,
    async_sessionmaker
)
import pandas as pd
from t2sql.utils import logger
import snowflake
import redshift_connector
import contextlib


class BaseDatabaseClient(ABC):
    def __init__(
        self,
        user: str,
        password: str,
        host: str,
        port: int,
        database: str,
        schema: str | None = None,
    ):
        self.user = user
        self.password = password
        self.host = host
        self.database = database
        self.port = port
        self.schema = schema
        self.engine = self._create_engine()

    @abstractmethod
    def _create_engine(self):
        """Creates and returns a SQLAlchemy engine."""
        pass

    def execute_query(self, query: str) -> pd.DataFrame | None:
        """Executes a query and returns results."""
        if not self.engine:
            raise ConnectionError("Database connection is not established.")

        try:
            with self.engine.cursor() as connection:
                connection.execute(query)
                df = pd.DataFrame(
                    connection.fetchall(),
                    columns=[desc[0] for desc in connection.description],
                )
                return df
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise e

    def close(self):
        """Closes the database connection."""
        if self.engine:
            self.engine.close()


class DatabaseSessionManager:
    def __init__(self, host: str, engine_kwargs=None):
        if engine_kwargs is None:
            engine_kwargs = {"future": True}
        self._engine = create_async_engine(host, **engine_kwargs)
        self._sessionmaker = async_sessionmaker(autocommit=False, bind=self._engine)

    @property
    def engine(self):
        return self._engine

    async def close(self) -> None:
        if self._engine is None:
            raise Exception("DatabaseSessionManager is not initialized")

        await self._engine.dispose()
        self._engine = None
        self._sessionmaker = None

    @contextlib.asynccontextmanager
    async def connect(self) -> AsyncConnection:
        if self._engine is None:
            raise Exception("DatabaseSessionManager is not initialized")

        async with self._engine.begin() as connection:
            try:
                yield connection
            except Exception:
                await connection.rollback()
                raise

    @contextlib.asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        if self._sessionmaker is None:
            raise Exception("DatabaseSessionManager is not initialized")

        session = self._sessionmaker()
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


class BaseAsyncDatabaseClient(BaseDatabaseClient):
    """Abstract base class for all database clients."""

    def __init__(
        self, user: str, password: str, host: str, port: int, database: str, schema: str
    ):
        super().__init__(user, password, host, port, database, schema)
        self.engine: DatabaseSessionManager | None = self._create_engine()

    @abstractmethod
    def _create_engine(self) -> DatabaseSessionManager:
        """Creates and returns a SQLAlchemy engine."""
        pass

    async def execute_query(self, query: str) -> pd.DataFrame | None:
        """Executes a query and returns results."""
        if not self.engine:
            raise ConnectionError("Database connection is not established.")

        try:
            async with self.engine.session() as session:
                result = await session.execute(text(query))
                df = pd.DataFrame(
                    result.fetchall(),
                )
                return df
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise e

    async def close(self):
        """Closes the database connection."""
        if self.engine:
            await self.engine.close()


class PostgresClient(BaseAsyncDatabaseClient):
    """PostgreSQL database client."""

    def _create_engine(self) -> DatabaseSessionManager:
        return DatabaseSessionManager(
            f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        )

    async def execute_query(self, query: str) -> pd.DataFrame | None:
        return await super().execute_query(query)


class MysqlClient(BaseAsyncDatabaseClient):
    """MySQL database client."""

    def _create_engine(self) -> DatabaseSessionManager:
        return DatabaseSessionManager(
            f"myssql+asyncmy://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        )


class MssqlClient(BaseAsyncDatabaseClient):
    """MsSQL database client."""

    def _create_engine(self) -> DatabaseSessionManager:
        return DatabaseSessionManager(
            f"mssql+aioodbc://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        )


class RedshiftClient(BaseDatabaseClient):
    """Redshift database client."""

    def __init__(
        self,
        user: str,
        password: str,
        host: str,
        port: int,
        database: str,
        schema: str | None = None,
    ):
        super().__init__(user, password, host, port, database, schema)
        self.engine = self._create_engine()

    def _create_engine(self):
        conn = redshift_connector.connect(
            database=self.database,
            password=self.password,
            user=self.user,
            host=self.host,
            port=self.port,
        )
        conn.autocommit = True
        return conn

    # TODO explore async version
    async def execute_query(self, query: str) -> pd.DataFrame | None:
        return await asyncio.to_thread(super().execute_query, query)


class SnowflakeClient(BaseDatabaseClient):
    """Snowflake database client."""

    def __init__(
        self,
        user: str,
        password: str,
        host: str,
        port: int,
        database: str,
        schema: str,
        warehouse: str,
        role: str,
    ):
        super().__init__(user, password, host, port, database)
        self.schema = schema
        self.warehouse = warehouse
        self.role = role

    def _create_engine(self):
        return snowflake.connector.connect(
            user=self.user,
            password=self.password,
            account=self.host,
            database=self.database,
            warehouse=self.warehouse,
            schema=self.schema,
            autocommit=True
        )

    # TODO explore async version
    async def execute_query(self, query: str) -> pd.DataFrame | None:
        return await asyncio.to_thread(super().execute_query, query)


class DatabaseClientFactory:
    """Factory to create database clients dynamically."""

    @staticmethod
    def create_client(source: str, connection_config: dict) -> BaseDatabaseClient:
        """Returns the appropriate database client based on the source type."""
        source = source.lower()

        if source == "postgres":
            return PostgresClient(**connection_config)
        elif source == "myssql":
            return MysqlClient(**connection_config)
        elif source == "mssql":
            return MssqlClient(**connection_config)
        elif source == "redshift":
            return RedshiftClient(**connection_config)
        elif source == "snowflake":
            return SnowflakeClient(**connection_config)
        else:
            raise ValueError(f"Unsupported database source: {source}")
