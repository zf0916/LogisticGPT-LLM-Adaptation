from logging import getLogger
from t2sql.ingestors.text_document_ingestor import TextIngestion


logger = getLogger(__name__)


async def index_schema(agent: TextIngestion) -> bool:
    logger.debug(
        f"Ingesting schema of the knowledgebase of agent {agent._descriptor_folder}."
    )
    try:
        await agent.train_on_information_schema(train_if_doc_exists=False)
    except Exception as e:
        logger.error(
            f"Cannot index schema of the knowledgebase of agent {agent._descriptor_folder}."
            f"Error: {str(e)}"
        )
        return False
    return True


async def ingest_text_file(
    document_name: str, document_data: str, agent: TextIngestion
) -> bool:
    logger.debug(
        f"Ingesting text file '{document_name}' to the knowledgebase of agent {agent._descriptor_folder}."
    )
    try:
        structured_doc = await agent.learn_md_document(document_name, document_data)
        await agent.learn_json_document(structured_doc, allow_replace=True)
    except Exception as e:
        logger.error(
            f"Cannot ingest text file '{document_name}' to the knowledgebase of agent {agent._descriptor_folder}."
            f"Error: {str(e)}"
        )
        return False
    return True


async def delete_text_files(doc_name: str, agent: TextIngestion) -> None:
    logger.debug(
        f"Deleting documents with name {doc_name} of the knowledgebase of agent {agent._descriptor_folder}."
    )
    await agent.remove_document(doc_name)


async def ingest_example(question: str, sql_code: str, agent: TextIngestion) -> bool:
    logger.debug(
        f"Ingesting example with question '{question}' and sql '{sql_code}' to the knowledgebase of agent {agent._descriptor_folder}."
    )
    try:
        await agent.remove_example(question)
        await agent.learn_sql(question, sql_code)
        await agent.expand_example_structure({"question": question, "sql": sql_code})
        await agent.train_in_domain_specific()
    except Exception as e:
        logger.error(
            f"Cannot ingest example with question '{question}' and  sql '{sql_code}' to the knowledgebase of agent {agent._descriptor_folder}."
            f"Error: {str(e)}"
        )
        return False
    return True


def update_business_rules(rules: list[str], agent: TextIngestion) -> bool:
    logger.debug(
        f"Updating business rules of the knowledgebase of agent {agent._descriptor_folder}."
    )
    try:
        agent.update_business_rules(rules)
    except Exception as e:
        logger.error(
            f"Cannot update business rules of the knowledgebase of agent {agent._descriptor_folder}."
            f"Error: {str(e)}"
        )
        return False
    return True


def update_prompts(sql_instruction: str, agent: TextIngestion) -> bool:
    logger.debug(
        f"Updating business rules of the knowledgebase of agent {agent._descriptor_folder}."
    )
    try:
        agent.update_prompts(sql_instruction)
    except Exception as e:
        logger.error(
            f"Cannot update sql instruction of the knowledgebase of agent {agent._descriptor_folder}."
            f"Error: {str(e)}"
        )
        return False
    return True


def load_examples(agent: TextIngestion) -> list[dict]:
    logger.debug(
        f"Loading examples of the knowledgebase of agent {agent._descriptor_folder}."
    )

    try:
        return agent.load_examples()
    except Exception as e:
        logger.error(
            f"Cannot get examples of the knowledgebase of agent {agent._descriptor_folder}."
            f"Error: {str(e)}"
        )
    return []


async def delete_example(question: str, agent: TextIngestion) -> None:
    logger.debug(
        f"Deleting example with question {question} of the knowledgebase of agent {agent._descriptor_folder}."
    )
    await agent.remove_example(question)
    await agent.train_in_domain_specific()


async def get_documentation(agent: TextIngestion):
    logger.debug(
        f"Getting docs  of the knowledgebase of agent {agent._descriptor_folder}."
    )
    result = await agent.get_all_documentation()
    return result


async def train_local(
    descriptor_base_path: str | None = None, train_if_doc_exists: bool = False
) -> None:
    from t2sql.agent import get_sql_agent

    agent = get_sql_agent(descriptor_base_path)
    await agent.generate_json_from_md_documentation()
    await agent.train_on_information_schema(train_if_doc_exists=train_if_doc_exists)
    await agent.train_on_documentation_json()
    await agent.expand_examples_structure()
    await agent.train_in_domain_specific()
    await agent.train_on_examples()
