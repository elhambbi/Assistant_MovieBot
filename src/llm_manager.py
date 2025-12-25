import sys
from pathlib import Path
from typing import Union

try:
    from vllm import AsyncLLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

from langchain_ollama import OllamaLLM

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import LLM_BACKEND, MODEL_NAME, VLLM_MODEL_HF, MAX_TOKENS, TEMPERATURE, logger


class LLMManager:
    def __init__(self, backend: str = None):
        self.backend = backend or LLM_BACKEND
        self.llm = None
        self.is_async = False
        
        if self.backend == "vllm" and VLLM_AVAILABLE:
            logger.info(f"Initializing vLLM backend with model: {VLLM_MODEL_HF}")
            try:
                self.llm = AsyncLLM(model=VLLM_MODEL_HF, dtype="float16", device="cuda")
                self.sampling_params = SamplingParams(
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE
                )
                self.is_async = True
            except Exception as e:
                logger.warning(f"vLLM initialization failed: {e}. Falling back to Ollama.")
                self._init_ollama()
        else:
            self._init_ollama()
        
        logger.info(f"LLMManager initialized with backend={self.backend}, is_async={self.is_async}")

    def _init_ollama(self):
        logger.info(f"Initializing Ollama backend with model: {MODEL_NAME}")
        self.llm = OllamaLLM(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            num_predict=MAX_TOKENS
        )
        self.backend = "ollama"
        self.is_async = False

    async def generate(self, prompt: str) -> str:
        if self.is_async:
            completion = await self.llm.completion(prompt, self.sampling_params)
            return completion.completions[0].text.strip()
        else:
            # For sync (Ollama), this is a blocking call
            return self.llm.invoke(prompt).strip()