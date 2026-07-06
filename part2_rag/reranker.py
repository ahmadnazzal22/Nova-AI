import re
import time
import numpy as np
from typing import Any
from .logger import get_logger

logger = get_logger(__name__)

_SCORE_THRESHOLD = 0.15
_TITLE_KEYWORDS_BONUS = 0.25
_KEYWORD_MATCH_WEIGHT = 0.3
_SEMANTIC_WEIGHT = 0.7
_RECENCY_BONUS = 0.10
_SOURCE_AUTHORITY_BONUS = 0.10

_MMR_LAMBDA = 0.6

_STOP_WORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with",
    "by", "from", "as", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
}

_NOISE_PATTERNS = [
    re.compile(r"click\s+here", re.I),
    re.compile(r"subscribe", re.I),
    re.compile(r"sign\s+up|newsletter", re.I),
    re.compile(r"advertisement|sponsored", re.I),
    re.compile(r"cookie\s+policy|privacy\s+policy", re.I),
    re.compile(r"all\s+rights?\s+reserved", re.I),
    re.compile(r"terms?\s+of\s+service", re.I),
    re.compile(r"follow\s+us\s+on", re.I),
    re.compile(r"buy\s+now|shop\s+now", re.I),
]

_SOURCE_AUTHORITY: dict[str, float] = {
    "wikipedia": 0.9,
    "arxiv": 0.95,
    "github": 0.85,
    "stackoverflow": 0.8,
    "stackexchange": 0.8,
    "medium": 0.5,
    "blog": 0.4,
    "news": 0.6,
    "docs": 0.85,
    "documentation": 0.85,
    "tutorial": 0.6,
    "reddit": 0.3,
}

_NUMERIC_YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-4]\d|2025|2026|2027|2028)\b")


def _extract_keywords(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z]\w+(?:[-/]\w+)*", text.lower())
    return {w for w in words if len(w) > 2 and w not in _STOP_WORDS}


def _keyword_overlap(query_keywords: set[str], doc_keywords: set[str]) -> float:
    if not query_keywords:
        return 0.0
    return len(query_keywords & doc_keywords) / len(query_keywords)


def _title_relevance(query_keywords: set[str], title: str) -> float:
    if not title or not query_keywords:
        return 0.0
    title_lower = title.lower()
    matches = sum(1 for kw in query_keywords if kw in title_lower)
    return matches / len(query_keywords)


def _noise_penalty(text: str) -> float:
    text_lower = text.lower()
    penalty = 0.0
    for pattern in _NOISE_PATTERNS:
        if pattern.search(text_lower):
            penalty += 0.15
    return min(penalty, 1.0)


def _recency_bonus(doc: dict) -> float:
    text = doc.get("text", doc.get("content", doc.get("snippet", "")))
    years = _NUMERIC_YEAR_RE.findall(text)
    if years:
        year = max(int(y) for y in years)
        current = 2026
        if year >= current - 2:
            return _RECENCY_BONUS
        elif year >= current - 5:
            return _RECENCY_BONUS * 0.5
    return 0.0


def _source_authority(doc: dict) -> float:
    url = doc.get("url", "").lower()
    source = doc.get("source", "").lower()
    title = doc.get("title", "").lower()
    text = (url + " " + source + " " + title)
    bonus = 0.0
    for keyword, authority in _SOURCE_AUTHORITY.items():
        if keyword in text:
            bonus = max(bonus, authority * _SOURCE_AUTHORITY_BONUS)
    return bonus


def _reduce_redundancy(docs: list[dict]) -> list[dict]:
    seen_texts: set[int] = set()
    result = []
    for doc in docs:
        text = doc.get("content", doc.get("text", doc.get("snippet", "")))
        h = hash(text[:200].strip().lower())
        if h not in seen_texts:
            seen_texts.add(h)
            result.append(doc)
    return result


def _clean_doc_text(text: str) -> str:
    text = re.sub(r"\[(?!\d+\])[^\]]*\]", "", text)
    for pattern in _NOISE_PATTERNS:
        text = pattern.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _mmr_select(docs: list[tuple[float, dict, np.ndarray]], query_emb: np.ndarray, k: int) -> list[tuple[float, dict]]:
    if not docs:
        return []
    selected = []
    remaining = list(docs)

    while len(selected) < k and remaining:
        best_idx = -1
        best_score = -float("inf")
        for i, (rel_score, doc, emb) in enumerate(remaining):
            if selected:
                max_sim = max(float(emb @ selected_emb) for _, _, selected_emb in selected)
            else:
                max_sim = 0.0
            mmr_score = _MMR_LAMBDA * rel_score - (1 - _MMR_LAMBDA) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i
        if best_idx >= 0:
            rel_score, doc, emb = remaining.pop(best_idx)
            selected.append((rel_score, doc, emb))

    return [(s, d) for s, d, _ in selected]


class Reranker:
    def __init__(self, embeddings_model: Any, top_k: int = 5):
        self.embeddings = embeddings_model
        self.top_k = top_k

    def _prepare_docs(self, documents: list[dict]) -> list[dict]:
        cleaned = []
        for doc in documents:
            raw_text = doc.get("content", doc.get("text", doc.get("snippet", "")))
            if raw_text:
                doc["_clean_text"] = _clean_doc_text(raw_text)
                cleaned.append(doc)
        return _reduce_redundancy(cleaned)

    def _score_doc(self, doc: dict, query_keywords: set[str], query_emb: np.ndarray, query_norm: float) -> float:
        text = doc["_clean_text"]
        title = doc.get("title", "")
        content = doc.get("content", doc.get("text", doc.get("snippet", "")))

        doc_emb = self._embed_text(text)
        doc_norm = np.linalg.norm(doc_emb)
        if doc_norm > 1e-10:
            doc_emb = doc_emb / doc_norm
        semantic_score = float(query_emb @ doc_emb)

        doc_keywords = _extract_keywords(text)
        keyword_score = _keyword_overlap(query_keywords, doc_keywords)
        title_score = _title_relevance(query_keywords, title)
        noise = _noise_penalty(content)

        recency = _recency_bonus(doc)
        authority = _source_authority(doc)

        combined_score = (
            _SEMANTIC_WEIGHT * semantic_score
            + _KEYWORD_MATCH_WEIGHT * keyword_score
            + _TITLE_KEYWORDS_BONUS * title_score
            + recency
            + authority
            - noise
        )

        doc["_semantic_score"] = round(semantic_score, 4)
        doc["_keyword_score"] = round(keyword_score, 4)
        doc["_title_score"] = round(title_score, 4)
        doc["_recency_bonus"] = round(recency, 4)
        doc["_authority_bonus"] = round(authority, 4)
        doc["_confidence_score"] = round(
            max(0.0, min(1.0, (semantic_score + keyword_score) / 2 + 0.3)), 4
        )
        doc["_relevance_score"] = round(max(0.0, combined_score), 4)

        return combined_score

    def _rank_internal(self, query: str, documents: list[dict], k: int | None, use_mmr: bool = False) -> list[tuple[dict, float]]:
        k = k or self.top_k
        if not documents:
            return []

        cleaned_docs = self._prepare_docs(documents)
        if not cleaned_docs:
            return []

        query_keywords = _extract_keywords(query)
        query_emb = self._embed_text(query)
        query_norm = np.linalg.norm(query_emb)
        if query_norm > 1e-10:
            query_emb = query_emb / query_norm

        scored: list[tuple[float, dict, np.ndarray]] = []
        for doc in cleaned_docs:
            text = doc["_clean_text"]
            doc_emb = self._embed_text(text)
            doc_norm = np.linalg.norm(doc_emb)
            if doc_norm > 1e-10:
                doc_emb = doc_emb / doc_norm
            combined = self._score_doc(doc, query_keywords, query_emb, query_norm)
            doc["_score"] = round(combined, 4)
            scored.append((combined, doc, doc_emb))

        scored.sort(key=lambda x: (-x[0], x[1].get("title", "")))

        if use_mmr:
            selected = _mmr_select(scored, query_emb, k)
        else:
            selected = [(s, d) for s, d, _ in scored[:k]]

        logger.debug(
            "Reranked %d docs -> top %d (threshold=%.2f, mmr=%s)",
            len(cleaned_docs), len(selected), _SCORE_THRESHOLD, use_mmr,
        )
        return [(doc, score) for score, doc in selected]

    def rerank(self, query: str, documents: list[dict], k: int | None = None) -> list[dict]:
        results = self._rank_internal(query, documents, k, use_mmr=True)
        return [doc for doc, _ in results]

    def rank_with_scores(self, query: str, documents: list[dict], k: int | None = None) -> list[tuple[dict, float]]:
        return self._rank_internal(query, documents, k, use_mmr=False)

    def _expand_query(self, query: str) -> list[str]:
        variations = [query]
        keywords = _extract_keywords(query)
        if keywords:
            kw_query = " ".join(keywords)
            if kw_query != query:
                variations.append(kw_query)
        return variations

    def _embed_text(self, text: str) -> np.ndarray:
        try:
            vec = self.embeddings.embed_query(text)
            return np.array(vec, dtype=np.float32)
        except Exception as e:
            logger.warning("Embedding failed: %s", e)
            return np.zeros(self.embeddings.config.d_model, dtype=np.float32)


def enrich_sources_with_scoring(sources: list[dict]) -> list[dict]:
    for s in sources:
        s.setdefault("relevance_score", round(s.get("_relevance_score", s.get("score", 0.5)), 4))
        s.setdefault("confidence_score", round(s.get("_confidence_score", 0.5), 4))
        s.setdefault("source_type", s.get("source", "web"))
        text = s.get("text", s.get("snippet", s.get("content", "")))
        keywords = _extract_keywords(text)
        s.setdefault("highlight_keywords", sorted(keywords)[:10] if keywords else [])
        s.pop("_clean_text", None)
        s.pop("_score", None)
        s.pop("_semantic_score", None)
        s.pop("_keyword_score", None)
        s.pop("_title_score", None)
        s.pop("_recency_bonus", None)
        s.pop("_authority_bonus", None)
    return sources
