import hashlib
import json
import ast
import uuid
import logging
import os
import numpy as np
from datetime import datetime
from t2sql.prompts import DEFAULT_PROMPTS, DEFAULT_SQL_INSTRUCTIONS
import copy
from sshtunnel import SSHTunnelForwarder


"""Setup logging configuration."""
# Create logs directory if it doesn't exist
if not os.path.exists("logs"):
    os.makedirs("logs")

# Configure logging
logger = logging.getLogger("text2sql")
logger.setLevel(logging.INFO)

# Create file handler
log_filename = f"logs/text2sql_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

DEFAULT_DESCRIPTOR_FILE_NAME: str = "t2sql_descriptor.json"


DEFAULT_BUSINESS_RULES = [

]


def parse_json(text: str) -> dict | list | None:
    try:
        result = json.loads(text.split("```json")[-1].split("```")[0])
    except:
        try:
            result = ast.literal_eval(text.split("```json")[-1].split("```")[0])
        except:
            return None
    return result


def parse_code(text: str) -> str | None:
    try:
        result = (
            text.split("```sql")[-1]
            .split("```")[0]
            .replace("<code>", "")
            .replace("</code>", "")
        )
    except:
        try:
            result = ast.literal_eval(
                text.split("```sql")[-1]
                .split("```")[0]
                .replace("<code>", "")
                .replace("</code>", "")
            )
        except:
            return None
    return result


def deterministic_uuid(content: str | bytes) -> str:
    """Creates deterministic UUID on hash value of string or byte content.

    Args:
        content: String or byte representation of data.

    Returns:
        UUID of the content.
    """
    if isinstance(content, str):
        content_bytes = content.encode("utf-8")
    elif isinstance(content, bytes):
        content_bytes = content
    else:
        raise ValueError(f"Content type {type(content)} not supported !")

    hash_object = hashlib.sha256(content_bytes)
    hash_hex = hash_object.hexdigest()
    namespace = uuid.UUID("00000000-0000-0000-0000-000000000000")
    content_uuid = str(uuid.uuid5(namespace, hash_hex))

    return content_uuid


def load_examples(path: str) -> list[dict]:
    """Load examples JSON file."""
    files = os.listdir(path)
    results = []

    for filename in files:
        input_path = os.path.join(path, filename)

        with open(input_path, "r") as file:
            state = json.load(file)
            results.append({"question": state.get("question"), "sql": state.get("sql")})

    return results


def load_prompts(path: str) -> dict[str]:
    """Load prompts from JSON file."""
    filename = os.path.join(path, DEFAULT_DESCRIPTOR_FILE_NAME)
    try:
        with open(filename, "r", encoding="utf-8") as f:
            prompts = json.load(f).get("prompts")
            used_prompts = copy.deepcopy(DEFAULT_PROMPTS)
            for prompt_name, prompt in prompts.items():
                used_prompts[prompt_name] = prompt
            return used_prompts
    except Exception as e:
        logger.warning(f"Error loading prompts: {e}, loading default prompts")
        return DEFAULT_PROMPTS


def calculate_threshold(n: int) -> int:
    """
    Calculate the threshold for table filtering based on count.

    Args:
        n (int): Number of samples

    Returns:
        int: Calculated threshold value
    """
    if n > 6:
        return np.ceil(n / 2)
    elif n in [5, 6]:
        return np.floor(n / 2)
    elif n == 4:
        return 1
    return 0


def create_default_descriptor(descriptor_base_path: str) -> dict:
    descriptor = {
        "router_model_list": [
            {
                "model_name": os.getenv("AZURE_API_DEFAULT_MODEL", "gpt-4o-2024-11-20"),
                "litellm_params": {
                    "model": f"azure/{os.getenv('AZURE_API_DEFAULT_MODEL', 'gpt-4o-2024-11-20')}",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
            },
            {
                "model_name": "o3-mini",
                "litellm_params": {
                    "model": "o3-mini",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
            {
                "model_name": "o1-mini",
                "litellm_params": {
                    "model": "o1-mini",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            },
        ],
        "open_ai_key": os.getenv("OPENAI_API_KEY"),
        "model": os.getenv("AZURE_API_DEFAULT_MODEL", "gpt-4o-2024-11-20"),
        "descriptors_path": "descriptors/default",
        "docs_md_folder": "training_data_storage/md_docs",
        "docs_json_folder": "training_data_storage/json_docs",
        "examples_folder": "training_data_storage/examples",
        "examples_extended_folder": "training_data_storage/train_examples",
        "docs_ddl_folder": "training_data_storage/ddl_docs",
        "router_default_max_parallel_requests": 20,
        "router_default_num_retries": 3,
        "db_path": "vector_db_storage",
        "collection_metadata": {"hnsw:space": "cosine"},
        "n_results_sql": 15,
        "client": "persistent",
        "business_rules": DEFAULT_BUSINESS_RULES,
        "prompts": {"DEFAULT_SQL_INSTRUCTIONS": DEFAULT_SQL_INSTRUCTIONS},
        "db": {
            "source": "postgres",
            "connection_config": {
                "schema": "public",
                "password": "postgres",
                "host": "localhost",
                "database": "dvdrental",
                "user": "postgres",
                "port": 5433,
            },
        },
    }

    with open(
        os.path.join(descriptor_base_path, DEFAULT_DESCRIPTOR_FILE_NAME), "w"
    ) as f:
        f.write(json.dumps(descriptor))

    return descriptor


def get_config(descriptor_base_path: str | None = None):
    if descriptor_base_path is None:
        descriptor_base_path = "default"

        descriptors_folder = os.getenv("T2SQL_DESCRIPTORS_FOLDER", "descriptors")
        if not os.path.exists(descriptors_folder):
            os.mkdir(descriptors_folder)

        descriptor_base_path = os.path.join(descriptors_folder, descriptor_base_path)

    if not os.path.exists(descriptor_base_path):
        os.mkdir(descriptor_base_path)

    descriptor_path = os.path.join(descriptor_base_path, DEFAULT_DESCRIPTOR_FILE_NAME)
    if os.path.exists(descriptor_path):
        with open(descriptor_path, "r") as f:
            descriptor = json.loads(f.read())
    else:
        descriptor = create_default_descriptor(descriptor_base_path)

    if descriptor.get("ssh_tunnel"):
        tunnel = SSHTunnelForwarder(
            descriptor["ssh_tunnel"]["host"],
            ssh_username=descriptor["ssh_tunnel"]["username"],
            ssh_pkey=descriptor["ssh_tunnel"]["private_key_path"],
            remote_bind_address=(
                descriptor["db"]["connection_config"]["host"],
                descriptor["db"]["connection_config"]["port"],
            ),
        )
        tunnel.start()
        descriptor["db"]["connection_config"]["host"] = (
            "127.0.0.1"  # Connect to tunnel locally
        )
        descriptor["db"]["connection_config"]["port"] = tunnel.local_bind_port

    descriptor["descriptors_folder"] = descriptor_base_path

    return descriptor
