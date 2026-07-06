import re
import numpy as np
from typing import Any

from ..logger import get_logger

logger = get_logger(__name__)

_MMR_LAMBDA = 0.6
_STOP_WORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with",
    "by", "from", "as", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
}

_SEMANTIC_WEIGHT = 0.7
_KEYWORD_WEIGHT = 0.3
_RECENCY_BONUS = 0.10
_SOURCE_AUTHORITY_BONUS = 0.10

_SOURCE_AUTHORITY: dict[str, float] = {
    "wikipedia": 0.9,
    "arxiv": 0.95,
    "github": 0.85,
    "stackoverflow": 0.8,
    "docs": 0.85,
    "documentation": 0.85,
}

_NUMERIC_YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-4]\d|2025|2026|2027|2028)\b")


def _extract_keywords(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z]\w+(?:[-/]\w+)*", text.lower())
    return {w for w in words if len(w) > 2 and w not in _STOP_WORDS}


def _keyword_overlap(qk: set[str], dk: set[str]) -> float:
    return len(qk & dk) / len(qk) if qk else 0.0


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


class RerankerService:
    def __init__(self, embedding_fn: Any = None):
        self._embedding_fn = embedding_fn

    def _embed(self, text: str) -> np.ndarray:
        if self._embedding_fn:
            try:
                vec = self._embedding_fn(text)
                return np.array(vec, dtype=np.float32)
            except Exception:
                pass
        return np.zeros(384, dtype=np.float32)

    def _score(self, query_emb: np.ndarray, query_keywords: set[str], doc: dict) -> float:
        text = doc.get("content", doc.get("text", doc.get("snippet", "")))
        if not text:
            return 0.0

        doc_emb = self._embed(text)
        dn = np.linalg.norm(doc_emb)
        if dn > 1e-10:
            doc_emb = doc_emb / dn
        semantic = float(query_emb @ doc_emb)

        kw_score = _keyword_overlap(query_keywords, _extract_keywords(text))

        recency = _recency_bonus(doc)
        authority = _source_authority(doc)

        combined = (
            _SEMANTIC_WEIGHT * semantic
            + _KEYWORD_WEIGHT * kw_score
            + recency
            + authority
        )

        doc["_relevance_score"] = round(max(0.0, combined), 4)
        doc["_semantic_score"] = round(semantic, 4)
        doc["_keyword_score"] = round(kw_score, 4)
        doc["_recency_bonus"] = round(recency, 4)
        doc["_authority_bonus"] = round(authority, 4)
        doc["_confidence_score"] = round(
            max(0.0, min(1.0, (semantic + kw_score) / 2 + 0.3)), 4
        )

        return combined

    def rank(self, query: str, documents: list[dict], k: int = 3, use_mmr: bool = True) -> list[dict]:
        if not documents:
            return []

        query_keywords = _extract_keywords(query)
        query_emb = self._embed(query)
        qn = np.linalg.norm(query_emb)
        if qn > 1e-10:
            query_emb = query_emb / qn

        scored = []
        for doc in documents:
            combined = self._score(query_emb, query_keywords, doc)
            text = doc.get("content", doc.get("text", doc.get("snippet", "")))
            doc_emb = self._embed(text)
            dn = np.linalg.norm(doc_emb)
            if dn > 1e-10:
                doc_emb = doc_emb / dn
            scored.append((combined, doc, doc_emb))

        scored.sort(key=lambda x: -x[0])

        if use_mmr and len(scored) > 1:
            selected = []
            remaining = list(scored)
            while len(selected) < k and remaining:
                best_idx = -1
                best_score = -float("inf")
                for i, (rel, doc, emb) in enumerate(remaining):
                    max_sim = max((float(emb @ se) for _, _, se in selected), default=0.0)
                    mmr = _MMR_LAMBDA * rel - (1 - _MMR_LAMBDA) * max_sim
                    if mmr > best_score:
                        best_score = mmr
                        best_idx = i
                if best_idx >= 0:
                    _, doc, _ = remaining.pop(best_idx)
                    selected.append((best_score, doc))
            return [d for _, d in selected]

        return [d for _, d, _ in scored[:k]]

    def rank_with_scores(self, query: str, documents: list[dict], k: int = 3) -> list[tuple[dict, float]]:
        if not documents:
            return []

        query_keywords = _extract_keywords(query)
        query_emb = self._embed(query)
        qn = np.linalg.norm(query_emb)
        if qn > 1e-10:
            query_emb = query_emb / qn

        results = []
        for doc in documents:
            combined = self._score(query_emb, query_keywords, doc)
            results.append((doc, round(max(0.0, combined), 4)))

        results.sort(key=lambda x: -x[1])
        return results[:k]


_reranker_service: RerankerService | None = None


def get_reranker_service() -> RerankerService:
    global _reranker_service
    if _reranker_service is None:
        _reranker_service = RerankerService()
    return _reranker_service


def enrich_sources_with_scoring(sources: list[dict]) -> list[dict]:
    for s in sources:
        s.setdefault("relevance_score", round(s.get("_relevance_score", s.get("score", 0.5)), 4))
        s.setdefault("confidence_score", round(s.get("_confidence_score", 0.5), 4))
        s.setdefault("source_type", s.get("source", "web"))
        text = s.get("text", s.get("snippet", s.get("content", "")))
        keywords = _extract_keywords(text)
        s.setdefault("highlight_keywords", sorted(keywords)[:10] if keywords else [])
    return sources
