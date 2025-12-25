import os
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv


config_dir = Path(__file__).parent
load_dotenv(config_dir / ".env")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("moviebot")

# LLM
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")  # "ollama" or "vllm"
MODEL_NAME = os.getenv("MODEL_NAME", "gemma2:9b")  # Ollama model name
VLLM_MODEL_HF = os.getenv("VLLM_MODEL_HF", "google/gemma-2-9b")  # HuggingFace model ID
USE_GPU = os.getenv("USE_GPU", "False").lower() == "true"
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))

# Vector Database
DATA_PATH = os.getenv("DATA_PATH", "data")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "faiss_movie_index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "3"))
