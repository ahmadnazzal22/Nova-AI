import os, re, time, html, urllib.request, urllib.parse
from typing import Optional
from .web_search import WebSearch, SearchResult, _clean_wikitext
from .logger import get_logger

logger = get_logger(__name__)

WIKI_RAW = "https://en.wikipedia.org/w/index.php"
_USER_AGENT = "LiveRAG/2.0"
_CACHE_TTL = 3600


class WebLoader:
    def __init__(self, cache_ttl: int = _CACHE_TTL, max_results: int = 5):
        self._cache: dict[str, tuple[list[str], float]] = {}
        self._cache_ttl = cache_ttl
        self._web_search = WebSearch(max_results=max_results)
        self.max_results = max_results

    def _cache_get(self, key: str) -> Optional[list[str]]:
        if key in self._cache:
            data, ts = self._cache[key]
            if time.time() - ts < self._cache_ttl:
                return data
            del self._cache[key]
        return None

    def _cache_set(self, key: str, data: list[str]):
        self._cache[key] = (data, time.time())
        if len(self._cache) > 100:
            stale = [k for k, (_, ts) in self._cache.items() if time.time() - ts > self._cache_ttl]
            for k in stale:
                del self._cache[k]

    def search_wikipedia(self, query: str, limit: int = 3) -> list[str]:
        cached = self._cache_get(f"search:{query}")
        if cached:
            logger.info("Wiki search cache hit: %.60s", query)
            return cached

        params = urllib.parse.urlencode({
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
        })
        url = f"https://en.wikipedia.org/w/api.php?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        try:
            resp = urllib.request.urlopen(req, timeout=10)
            data = resp.read().decode("utf-8")
            import json
            j = json.loads(data)
            titles = [r["title"] for r in j.get("query", {}).get("search", [])]
            if titles:
                logger.info("Wiki search: %s -> %d results", query, len(titles))
            self._cache_set(f"search:{query}", titles)
            return titles
        except Exception as e:
            logger.warning("Wiki search failed: %s", e)
            return []

    def fetch_article(self, title: str, max_chars: int = 3000) -> str:
        cached = self._cache_get(f"article:{title}")
        if cached:
            return "\n\n".join(cached)

        safe = urllib.parse.quote(title.replace(" ", "_"))
        url = f"{WIKI_RAW}?action=raw&title={safe}"
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        try:
            resp = urllib.request.urlopen(req, timeout=10)
            raw = resp.read().decode("utf-8")
            cleaned = _clean_wikitext(raw)
            cleaned = re.sub(r"(?i)==\s*(see also|references|further reading|external links|notes|sources|bibliography)\s*==.*$", "", cleaned, flags=re.DOTALL)
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
            if len(cleaned) > max_chars:
                cleaned = cleaned[:max_chars]
            if len(cleaned) > 500:
                logger.info("Fetched article: %s (%d chars)", title, len(cleaned))
            self._cache_set(f"article:{title}", [cleaned])
            return cleaned
        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.warning("Article not found: %s", title)
            else:
                logger.warning("HTTP %d fetching %s", e.code, title)
            return ""
        except Exception as e:
            logger.warning("Fetch failed for %s: %s", title, e)
            return ""

    def fetch_for_query(self, query: str, max_articles: int = 3, max_chars: int = 5000) -> list[str]:
        cached = self._cache_get(f"query:{query}")
        if cached:
            logger.info("Live fetch cache hit: %.60s", query)
            return cached

        results = self._web_search.search_and_fetch(
            query,
            max_results=max_articles,
            max_chars_per_page=max_chars // max_articles,
        )

        texts = []
        for r in results:
            if r.content:
                texts.append(r.content)
            elif r.snippet:
                texts.append(r.snippet)

        combined = "\n\n".join(texts)
        if len(combined) > max_chars:
            combined = combined[:max_chars]

        result = [combined] if combined else []
        self._cache_set(f"query:{query}", result)
        return result

    def fetch_with_metadata(self, query: str, max_results: int = 5, max_chars_per_page: int = 2000) -> list[dict]:
        cached_key = f"meta:{query}:{max_results}"
        cached = self._cache_get(cached_key)
        if cached:
            return [eval(s) for s in cached] if cached else []

        results = self._web_search.search_and_fetch(
            query,
            max_results=max_results,
            max_chars_per_page=max_chars_per_page,
        )

        metadata = []
        for r in results:
            if r.content or r.snippet:
                metadata.append(r.to_dict())

        self._cache_set(cached_key, [str(m) for m in metadata])
        return metadata

    def fetch_and_chunk(self, query: str, max_results: int = 3, max_chars_total: int = 5000) -> list[dict]:
        raw_texts = self.fetch_for_query(query, max_articles=max_results, max_chars=max_chars_total)
        if not raw_texts:
            return []
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=60, length_function=len, separators=["\n\n", "\n", ".", " ", ""])
        chunks = splitter.split_text(raw_texts[0])
        chunks = [c.strip() for c in chunks if len(c.strip()) > 20]
        chunks = list(dict.fromkeys(chunks))
        return [{"text": c, "source": "web", "title": "", "url": ""} for c in chunks]
