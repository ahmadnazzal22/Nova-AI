import time
import uuid
import functools
import threading
from contextlib import contextmanager
from typing import Any, Callable

from ..logger import get_logger, set_correlation_id

logger = get_logger(__name__)

_trace_context = threading.local()


def get_trace_id() -> str:
    if not hasattr(_trace_context, "trace_id") or _trace_context.trace_id is None:
        _trace_context.trace_id = uuid.uuid4().hex[:16]
    return _trace_context.trace_id


def set_trace_id(trace_id: str | None = None):
    _trace_context.trace_id = trace_id or uuid.uuid4().hex[:16]
    set_correlation_id(_trace_context.trace_id)


@contextmanager
def trace_span(operation: str, attributes: dict | None = None):
    trace_id = get_trace_id()
    span_id = uuid.uuid4().hex[:8]
    start = time.perf_counter()
    try:
        yield
        elapsed = time.perf_counter() - start
        logger.debug("[TRACE %s/%s] %s completed in %.2fms", trace_id, span_id, operation, elapsed * 1000)
    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.warning("[TRACE %s/%s] %s FAILED after %.2fms: %s", trace_id, span_id, operation, elapsed * 1000, e)
        raise


def traced(level: str = "DEBUG") -> Callable:
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with trace_span(func.__qualname__):
                return func(*args, **kwargs)
        return wrapper
    return decorator
