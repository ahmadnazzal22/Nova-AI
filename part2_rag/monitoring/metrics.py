import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field

from ..logger import get_logger

logger = get_logger(__name__)


@dataclass
class MetricPoint:
    count: int = 0
    total_time: float = 0.0
    errors: int = 0
    last_seen: float = 0.0


class MetricsCollector:
    def __init__(self):
        self._lock = threading.Lock()
        self._metrics: dict[str, MetricPoint] = defaultdict(MetricPoint)
        self._cache_hits: dict[str, int] = defaultdict(int)
        self._cache_misses: dict[str, int] = defaultdict(int)

    def record(self, name: str, duration: float = 0.0, error: bool = False):
        with self._lock:
            m = self._metrics[name]
            m.count += 1
            m.total_time += duration
            if error:
                m.errors += 1
            m.last_seen = time.time()

    def record_cache(self, name: str, hit: bool):
        with self._lock:
            if hit:
                self._cache_hits[name] += 1
            else:
                self._cache_misses[name] += 1

    def get_metrics(self) -> dict:
        with self._lock:
            result = {}
            for name, m in self._metrics.items():
                avg_time = m.total_time / m.count if m.count else 0
                result[name] = {
                    "count": m.count,
                    "avg_time_ms": round(avg_time * 1000, 2),
                    "total_time_ms": round(m.total_time * 1000, 2),
                    "errors": m.errors,
                    "error_rate": round(m.errors / m.count * 100, 2) if m.count else 0,
                    "last_seen": m.last_seen,
                }
            result["_cache"] = {
                name: {"hits": h, "misses": self._cache_misses.get(name, 0), "hit_rate": round(h / (h + self._cache_misses.get(name, 1)) * 100, 1)}
                for name, h in self._cache_hits.items()
            }
            return result

    def avg_latency_ms(self, name: str) -> float:
        with self._lock:
            m = self._metrics.get(name)
            if m and m.count:
                return (m.total_time / m.count) * 1000
            return 0.0


_metrics: MetricsCollector | None = None


def get_metrics() -> MetricsCollector:
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics
