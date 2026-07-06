import os
import json
import time
import threading
from typing import Generator, Any
from dataclasses import dataclass

from ..logger import get_logger

logger = get_logger(__name__)

PREFERRED_MODELS = ["phi3:mini", "phi4", "llama3.2", "gemma4", "gemma3", "llama3.2:1b", "tinyllama", "mistral"]
WEAK_MODELS = frozenset({"llama3.2:1b", "tinyllama"})


@dataclass
class ModelTier:
    name: str
    tier: str  # "strong" | "weak" | "mock"


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 60.0):
        self._failures = 0
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._last_failure_time = 0.0
        self._state = "closed"  # closed | open | half-open
        self._lock = threading.Lock()

    def record_failure(self):
        with self._lock:
            self._failures += 1
            self._last_failure_time = time.time()
            if self._failures >= self._failure_threshold:
                self._state = "open"
                logger.warning("Circuit breaker OPEN (failures=%d)", self._failures)

    def record_success(self):
        with self._lock:
            self._failures = 0
            self._state = "closed"

    @property
    def is_open(self) -> bool:
        with self._lock:
            if self._state == "open":
                if time.time() - self._last_failure_time > self._recovery_timeout:
                    self._state = "half-open"
                    logger.info("Circuit breaker HALF-OPEN (attempting recovery)")
                    return False
                return True
            return False

    def reset(self):
        with self._lock:
            self._failures = 0
            self._state = "closed"


class LLMService:
    def __init__(self, ollama_url: str = "http://localhost:11434", groq_key: str = "", timeout: int = 60):
        self._ollama_url = ollama_url
        self._groq_key = groq_key
        self._timeout = timeout
        self._llm = None
        self._llm_lock = threading.Lock()
        self._circuit_breaker = CircuitBreaker()

    def _get_ollama_models(self) -> list[dict]:
        import httpx
        try:
            r = httpx.get(f"{self._ollama_url}/api/tags", timeout=5)
            r.raise_for_status()
            return r.json().get("models", [])
        except Exception as e:
            logger.warning("Ollama unavailable: %s", e)
            return []

    def _select_model(self, models: list[dict], max_size_gb: float = 5.0) -> ModelTier:
        small_models = [m for m in models if m.get("size", 0) < max_size_gb * 1_000_000_000]

        def normalize(name: str) -> str:
            return name.removesuffix(":latest")

        available = {normalize(m["name"]) for m in small_models}

        strong_available = {m for m in available if m not in WEAK_MODELS}
        preferred_strong = [m for m in PREFERRED_MODELS if m in (strong_available if strong_available else available)]

        for pref in preferred_strong:
            if pref in available:
                return ModelTier(name=pref, tier="strong" if pref not in WEAK_MODELS else "weak")

        if small_models:
            smallest = min(small_models, key=lambda m: m.get("size", 0))
            tier = "weak" if normalize(smallest["name"]) in WEAK_MODELS else "strong"
            return ModelTier(name=smallest["name"], tier=tier)

        return ModelTier(name="", tier="mock")

    def get_llm(self) -> Any:
        if self._llm is not None:
            return self._llm
        with self._llm_lock:
            if self._llm is not None:
                return self._llm
            self._llm = self._build_llm()
            return self._llm

    def _build_llm(self) -> Any:
        if self._groq_key:
            return self._build_groq_llm()
        models = self._get_ollama_models()
        if not models:
            logger.warning("No LLM available, returning MockLLM")
            return MockLLM()
        selected = self._select_model(models)
        if selected.tier == "mock":
            logger.warning("No suitable model found, returning MockLLM")
            return MockLLM()
        logger.info("Selected LLM: %s (tier=%s)", selected.name, selected.tier)
        from langchain_ollama import OllamaLLM
        return OllamaLLM(model=selected.name, temperature=0.3, timeout=self._timeout)

    def _build_groq_llm(self) -> Any:
        from .groq_llm import GroqLLM
        return GroqLLM(model="llama-3.1-8b-instant", temperature=0.3)

    def invoke(self, prompt: str, fallback_prompt: str | None = None) -> str:
        if self._circuit_breaker.is_open:
            logger.warning("Circuit breaker open, raising error")
            raise RuntimeError("LLM circuit breaker open — service temporarily unavailable")
        try:
            llm = self.get_llm()
            result = llm.invoke(prompt)
            self._circuit_breaker.record_success()
            return result
        except Exception as e:
            self._circuit_breaker.record_failure()
            logger.error("LLM invoke failed: %s", e)
            raise

    def stream(self, prompt: str) -> Generator[str, None, None]:
        if self._circuit_breaker.is_open:
            raise RuntimeError("LLM circuit breaker open — service temporarily unavailable")
        try:
            llm = self.get_llm()
            for token in llm.stream(prompt):
                yield token if isinstance(token, str) else (token.text if hasattr(token, 'text') else str(token))
            self._circuit_breaker.record_success()
        except Exception as e:
            self._circuit_breaker.record_failure()
            logger.error("LLM stream failed: %s", e)
            raise


class MockLLM:
    """Last-resort fallback when no real LLM is available.
    Raises an exception so callers know the pipeline is degraded,
    rather than silently returning static text."""

    def invoke(self, prompt: str, **kwargs) -> str:
        raise RuntimeError("No LLM available — check Ollama or Groq configuration. "
                           "MockLLM cannot generate real answers.")

    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        raise RuntimeError("No LLM available — check Ollama or Groq configuration. "
                           "MockLLM cannot generate real answers.")


_llm_service: LLMService | None = None


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        from ..config.settings import settings
        _llm_service = LLMService(
            ollama_url=settings.ollama_base_url,
            groq_key=settings.groq_api_key,
            timeout=settings.ollama_timeout,
        )
    return _llm_service
