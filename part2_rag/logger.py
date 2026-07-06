import os
import time
import functools
import logging
from typing import Callable, Any

from part1_transformer.logger import get_logger as get_base_logger, LOG_FORMAT, DATE_FORMAT

__all__ = ["get_logger", "LOG_FORMAT", "DATE_FORMAT", "timed", "log_pipeline"]


# Extended log format with correlation ID support
RAG_LOG_FORMAT = "%(asctime)s | %(levelname)8s | %(name)s | %(correlation_id)s | %(message)s"


class CorrelationFilter(logging.Filter):
    """Add correlation_id to all log records."""
    _correlation_id: str = ""

    @classmethod
    def set_correlation_id(cls, cid: str):
        cls._correlation_id = cid

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = self._correlation_id or "-"
        return True


# Configure the RAG logger once
_rag_logger = logging.getLogger("rag")
_rag_logger.propagate = False
if not _rag_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(RAG_LOG_FORMAT))
    _rag_logger.addHandler(_handler)
    _rag_logger.addFilter(CorrelationFilter())

_log_level = os.getenv("RAG_LOG_LEVEL", os.getenv("LOG_LEVEL", "INFO")).upper()
_rag_logger.setLevel(getattr(logging, _log_level, logging.INFO))


def get_logger(name: str) -> logging.Logger:
    """Get a logger with RAG-specific configuration."""
    logger = logging.getLogger(f"rag.{name}")
    logger.propagate = False
    if not logger.handlers:
        logger.addHandler(_rag_logger.handlers[0])
        logger.addFilter(CorrelationFilter())
    logger.setLevel(_rag_logger.level)
    return logger


def set_correlation_id(cid: str):
    CorrelationFilter.set_correlation_id(cid)


def timed(level: str = "DEBUG") -> Callable:
    """Decorator that logs the execution time of a function."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger = get_logger(func.__module__)
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.log(getattr(logging, level.upper(), logging.DEBUG),
                           "%s took %.3fs", func.__qualname__, elapsed)
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                logger.warning("%s failed after %.3fs: %s", func.__qualname__, elapsed, e)
                raise
        return wrapper
    return decorator


def log_pipeline(stage: str, details: dict, logger: logging.Logger):
    """Log a structured pipeline stage update."""
    parts = [f"[{stage}]"]
    for k, v in details.items():
        parts.append(f"{k}={v}")
    logger.info(" ".join(parts))
