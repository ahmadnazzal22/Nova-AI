import hashlib
import json
import time
from typing import Any

from ..logger import get_logger

logger = get_logger(__name__)


class RedisCache:
    def __init__(self, redis_url: str = "redis://localhost:6379/0", default_ttl: int = 300, semantic_threshold: float = 0.92):
        self._client = None
        self._redis_url = redis_url
        self._default_ttl = default_ttl
        self._semantic_threshold = semantic_threshold
        self._local_cache: dict[str, tuple[Any, float]] = {}
        self._enabled = True

    @property
    def client(self):
        if self._client is None:
            try:
                import redis.asyncio as aioredis
                self._client = aioredis.from_url(self._redis_url, decode_responses=True)
                logger.info("Redis cache connected: %s", self._redis_url)
            except Exception as e:
                logger.warning("Redis unavailable, using local cache fallback: %s", e)
                self._enabled = False
        return self._client

    def _make_key(self, prefix: str, data: str) -> str:
        h = hashlib.sha256(data.encode()).hexdigest()[:16]
        return f"rag:{prefix}:{h}"

    async def get(self, prefix: str, key: str) -> Any | None:
        cache_key = self._make_key(prefix, key)
        if self._enabled:
            try:
                val = await self.client.get(cache_key)
                if val:
                    return json.loads(val)
            except Exception as e:
                logger.debug("Cache get failed: %s", e)
        entry = self._local_cache.get(cache_key)
        if entry:
            val, ts = entry
            if time.time() - ts < self._default_ttl:
                return val
            del self._local_cache[cache_key]
        return None

    async def set(self, prefix: str, key: str, value: Any, ttl: int | None = None):
        cache_key = self._make_key(prefix, key)
        ttl = ttl or self._default_ttl
        if self._enabled:
            try:
                await self.client.setex(cache_key, ttl, json.dumps(value, default=str))
                return
            except Exception as e:
                logger.debug("Cache set failed: %s", e)
        self._local_cache[cache_key] = (value, time.time())

    async def delete(self, prefix: str, key: str):
        cache_key = self._make_key(prefix, key)
        self._local_cache.pop(cache_key, None)
        if self._enabled:
            try:
                await self.client.delete(cache_key)
            except Exception:
                pass

    async def flush_prefix(self, prefix: str):
        pattern = f"rag:{prefix}:*"
        if self._enabled:
            try:
                cursor = 0
                while True:
                    cursor, keys = await self.client.scan(cursor, match=pattern, count=100)
                    if keys:
                        await self.client.delete(*keys)
                    if cursor == 0:
                        break
            except Exception:
                pass
        self._local_cache = {k: v for k, v in self._local_cache.items() if not k.startswith(f"rag:{prefix}:")}


_query_cache: RedisCache | None = None


def get_query_cache() -> RedisCache:
    global _query_cache
    if _query_cache is None:
        from ..config.settings import settings
        _query_cache = RedisCache(redis_url=settings.redis_url, default_ttl=settings.redis_cache_ttl, semantic_threshold=settings.redis_semantic_threshold)
    return _query_cache
