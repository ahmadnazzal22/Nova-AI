import os
import re
import time
import html
import warnings
import urllib.request
import urllib.parse
from typing import Optional
from .logger import get_logger

warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")
logger = get_logger(__name__)

_USER_AGENT = "LiveRAG/2.0"
_SEARCH_CACHE_TTL = 600
_MAX_RESULTS = 5

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_RAW = "https://en.wikipedia.org/w/index.php"

# ── Query Rewriting ───────────────────────────────────────────────
_FILLER_WORDS = {
    "can", "could", "would", "should", "tell", "give", "what", "how",
    "why", "when", "where", "who", "which", "do", "does", "did", "is",
    "are", "was", "were", "please", "explain", "describe", "define",
    "about", "the", "a", "an", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "that", "this", "these", "those", "it", "its",
    "you", "your", "me", "my", "i", "we", "our", "they", "their",
}

_TECHNICAL_EXPANSIONS = {
    "attention": "self attention mechanism transformer neural network",
    "transformer": "transformer model architecture deep learning",
    "llm": "large language model",
    "nlp": "natural language processing",
    "cnn": "convolutional neural network",
    "rnn": "recurrent neural network",
    "lstm": "long short term memory",
    "gan": "generative adversarial network",
    "bert": "bert model language representation",
    "gpt": "gpt model generative pre trained transformer",
    "nn": "neural network",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "rag": "retrieval augmented generation",
    "api": "application programming interface",
    "sse": "server sent events",
    "jwt": "json web token authentication",
    "orm": "object relational mapping",
    "sql": "structured query language database",
    "regex": "regular expression pattern matching",
}

_NOISE_DOMAINS = {
    "youtube.com", "youtu.be", "twitch.tv", "vimeo.com", "dailymotion.com",
    "netflix.com", "hulu.com", "disneyplus.com", "hbomax.com",
    "espn.com", "skysports.com", "sports.yahoo.com", "goal.com",
    "transfermarkt.com", "livescore.com", "flashscore.com",
    "imdb.com", "rottentomatoes.com", "metacritic.com",
    "doubleclick.net", "googleadservices.com", "amazon.com",
    "ebay.com", "etsy.com", "walmart.com", "alibaba.com", "aliexpress.com",
    "facebook.com", "instagram.com", "tiktok.com", "snapchat.com",
    "reddit.com", "twitter.com", "x.com", "linkedin.com", "pinterest.com",
    "tripadvisor.com", "yelp.com", "zillow.com",
    "cricbuzz.com", "espncricinfo.com",
}

_TRUSTED_DOMAINS = {
    "wikipedia.org", "github.com", "arxiv.org", "acm.org",
    "ieee.org", "springer.com", "sciencedirect.com",
    "nature.com", "science.org", "plos.org",
    "pytorch.org", "tensorflow.org", "docs.python.org",
    "readthedocs.io", "mozilla.org", "developer.mozilla.org",
    "stackoverflow.com", "stackexchange.com",
    "medium.com", "towardsdatascience.com", "analyticsvidhya.com",
    "geeksforgeeks.org", "tutorialspoint.com",
    "oracle.com", "microsoft.com", "google.com", "ibm.com",
    "nginx.com", "docker.com", "kubernetes.io",
    "fastapi.tiangolo.com", "streamlit.io",
}


def rewrite_query(question: str) -> str:
    words = question.lower().split()
    filtered = [w for w in words if w not in _FILLER_WORDS and len(w) > 1]
    if not filtered:
        filtered = words
    expanded = []
    for w in filtered:
        if w in _TECHNICAL_EXPANSIONS:
            expanded.append(_TECHNICAL_EXPANSIONS[w])
        else:
            expanded.append(w)
    return " ".join(expanded)


def filter_domain(url: str) -> bool:
    if not url:
        return True
    try:
        domain = urllib.parse.urlparse(url).netloc.lower()
        domain = re.sub(r"^www\.", "", domain)
        if domain in _TRUSTED_DOMAINS:
            return True
        for noise in _NOISE_DOMAINS:
            if noise in domain or domain in noise:
                logger.debug("Filtered noise domain: %s", domain)
                return False
        return True
    except Exception:
        return True


class SearchResult:
    def __init__(self, title: str, url: str, snippet: str, content: str = "", source: str = "web"):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.content = content
        self.source = source

    def to_dict(self) -> dict:
        return {"title": self.title, "url": self.url, "snippet": self.snippet, "content": self.content, "source": self.source}


def _clean_wikitext(text: str) -> str:
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[\[File:.*?\]\]", "", text, flags=re.DOTALL)
    text = re.sub(r"\[\[Image:.*?\]\]", "", text, flags=re.DOTALL)
    text = re.sub(r"\[\[Category:[^\]]*\]\]", "", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"={2,}.*?={2,}", ". ", text)
    text = html.unescape(text)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _truncate(text: str, max_chars: int = 300) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


class WebSearch:
    def __init__(self, max_results: int = _MAX_RESULTS, cache_ttl: int = _SEARCH_CACHE_TTL):
        self.max_results = max_results
        self.cache_ttl = cache_ttl
        self._cache: dict[str, tuple[list[SearchResult], float]] = {}
        self._duckduckgo_available = None

    def _cache_get(self, key: str) -> Optional[list[SearchResult]]:
        if key in self._cache:
            data, ts = self._cache[key]
            if time.time() - ts < self.cache_ttl:
                return data
            del self._cache[key]
        return None

    def _cache_set(self, key: str, data: list[SearchResult]):
        self._cache[key] = (data, time.time())
        if len(self._cache) > 100:
            stale = [k for k, (_, ts) in self._cache.items() if time.time() - ts > self.cache_ttl]
            for k in stale:
                del self._cache[k]

    def _check_duckduckgo(self) -> bool:
        if self._duckduckgo_available is not None:
            return self._duckduckgo_available
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                list(ddgs.text("test", max_results=5))
            self._duckduckgo_available = True
            logger.info("DuckDuckGo search available")
        except Exception as e:
            logger.warning("DuckDuckGo unavailable: %s", e)
            self._duckduckgo_available = False
        return self._duckduckgo_available

    def search_web(self, query: str) -> list[SearchResult]:
        rewritten = rewrite_query(query)
        if rewritten != query:
            logger.debug("Query rewritten: %.60s -> %.60s", query, rewritten)
        cache_key = f"web:{rewritten}"
        cached = self._cache_get(cache_key)
        if cached:
            logger.info("Web search cache hit: %.60s", rewritten)
            return cached

        results = []
        if self._check_duckduckgo():
            results = self._search_duckduckgo(rewritten)
        if not results:
            results = self._search_wikipedia(rewritten)
        if not results:
            logger.warning("All search backends returned 0 results for: %.60s", rewritten)

        filtered = [r for r in results if filter_domain(r.url)]

        n_blocked = len(results) - len(filtered)
        if n_blocked:
            logger.debug("Filtered %d noise domains from %d results", n_blocked, len(results))

        filtered = filtered[:self.max_results]
        self._cache_set(cache_key, filtered)
        logger.info("Web search: %.60s -> %d results (%d blocked)", rewritten, len(filtered), n_blocked)
        return filtered

    def _search_duckduckgo(self, query: str) -> list[SearchResult]:
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=self.max_results))
            results = []
            for r in raw:
                title = r.get("title", "")
                url = r.get("href", "")
                snippet = r.get("body", "")
                results.append(SearchResult(title=title, url=url, snippet=snippet, source="duckduckgo"))
            return results
        except Exception as e:
            logger.warning("DuckDuckGo search failed: %s", e)
            return self._search_wikipedia(query)

    def fetch_page(self, url: str, timeout: int = 10) -> tuple[str, str]:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            resp = urllib.request.urlopen(req, timeout=timeout)
            raw = resp.read().decode("utf-8", errors="replace")
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(raw, "lxml")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            title = soup.title.string.strip() if soup.title and soup.title.string else ""
            body = soup.get_text(separator="\n")
            body = re.sub(r"\n{3,}", "\n\n", body)
            body = re.sub(r"\s+", " ", body).strip()
            return title, body
        except Exception as e:
            logger.warning("Fetch page failed for %s: %s", url, e)
            return "", ""

    def _search_wikipedia(self, query: str) -> list[SearchResult]:
        params = urllib.parse.urlencode({
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": self.max_results,
            "format": "json",
        })
        url = f"{WIKI_API}?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        try:
            resp = urllib.request.urlopen(req, timeout=10)
            import json
            j = json.loads(resp.read().decode("utf-8"))
            results = []
            for r in j.get("query", {}).get("search", []):
                title = r["title"]
                snippet = re.sub(r"<[^>]+>", "", r.get("snippet", ""))
                page_id = r.get("pageid", "")
                url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                results.append(SearchResult(title=title, url=url, snippet=snippet, source="wikipedia"))
            return results
        except Exception as e:
            logger.warning("Wikipedia search failed: %s", e)
            return []

    def fetch_wikipedia_article(self, title: str, max_chars: int = 3000) -> str:
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
            logger.info("Fetched wiki article: %s (%d chars)", title, len(cleaned))
            return cleaned
        except Exception as e:
            logger.warning("Wiki fetch failed for %s: %s", title, e)
            return ""

    def search_and_fetch(self, query: str, max_results: int = 3, max_chars_per_page: int = 2000) -> list[SearchResult]:
        search_results = self.search_web(query)
        enriched = []
        for sr in search_results[:max_results]:
            if not filter_domain(sr.url):
                continue
            if sr.source == "wikipedia":
                content = self.fetch_wikipedia_article(sr.title, max_chars=max_chars_per_page)
            elif sr.url:
                _, content = self.fetch_page(sr.url)
                content = _truncate(content, max_chars_per_page)
            else:
                content = ""
            sr.content = content
            enriched.append(sr)
        return enriched
