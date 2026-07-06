"""
Context Compression Layer:
- Remove redundant chunks
- Merge similar passages
- Extract answer-critical sentences only
- Enforce token budget dynamically
"""
import re
import numpy as np
from typing import Any, Optional
from collections import Counter

from .logger import get_logger

logger = get_logger(__name__)

_MAX_CONTEXT_CHARS = 6000
_MIN_CHUNK_LENGTH = 50
_SIMILARITY_THRESHOLD = 0.85
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'\u0600-\u06FF])")


def _get_text(chunk: dict) -> str:
    return chunk.get("text", chunk.get("content", chunk.get("snippet", "")))


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z]\w+", text.lower()))


class ContextCompressor:
    def __init__(self, embeddings: Any = None, llm: Any = None):
        self._embeddings = embeddings
        self._llm = llm

    def compress(
        self,
        chunks: list[dict],
        query: str,
        max_chars: int = _MAX_CONTEXT_CHARS,
    ) -> list[dict]:
        if not chunks:
            return []

        query_keywords = _tokenize(query)

        deduped = self._deduplicate(chunks)

        merged = self._merge_similar(deduped, query_keywords)

        extracted = self._extract_critical(merged, query_keywords)

        compressed = self._enforce_budget(extracted, max_chars)

        logger.debug(
            "Context compression: %d chunks -> %d (%.0f%% reduction)",
            len(chunks), len(compressed),
            (1 - len(compressed) / max(1, len(chunks))) * 100,
        )
        return compressed

    def _deduplicate(self, chunks: list[dict]) -> list[dict]:
        seen_texts: set[int] = set()
        result = []
        for c in chunks:
            text = _get_text(c)
            if not text:
                continue
            key = hash(text[:300].strip().lower())
            if key in seen_texts:
                continue
            seen_texts.add(key)

            dup = False
            for existing in result:
                existing_text = _get_text(existing)
                if self._texts_are_near_duplicate(text, existing_text):
                    dup = True
                    self._merge_metadata(existing, c)
                    break
            if not dup:
                result.append(c)

        return result

    def _texts_are_near_duplicate(self, a: str, b: str) -> bool:
        if not a or not b:
            return False
        a_norm = a.lower().strip()
        b_norm = b.lower().strip()

        if len(a_norm) < 20 or len(b_norm) < 20:
            return a_norm == b_norm

        if a_norm == b_norm:
            return True

        a_tokens = set(a_norm.split())
        b_tokens = set(b_norm.split())
        intersection = a_tokens & b_tokens
        union = a_tokens | b_tokens
        if not union:
            return False
        jaccard = len(intersection) / len(union)
        if jaccard > 0.75:
            return True

        if len(a_norm) > 50 and len(b_norm) > 50:
            longer = a_norm if len(a_norm) >= len(b_norm) else b_norm
            shorter = b_norm if len(a_norm) >= len(b_norm) else a_norm
            if shorter in longer:
                return True

        return False

    def _merge_metadata(self, target: dict, source: dict):
        for key in ("title", "url", "source"):
            if not target.get(key) and source.get(key):
                target[key] = source[key]
        existing_score = target.get("_relevance_score", 0) or target.get("score", 0)
        new_score = source.get("_relevance_score", 0) or source.get("score", 0)
        target["_relevance_score"] = max(float(existing_score), float(new_score))

    def _merge_similar(
        self, chunks: list[dict], query_keywords: set[str]
    ) -> list[dict]:
        if len(chunks) < 2:
            return chunks

        result = [chunks[0]]
        for chunk in chunks[1:]:
            last = result[-1]
            last_text = _get_text(last)
            chunk_text = _get_text(chunk)

            if self._should_merge(last_text, chunk_text, query_keywords):
                merged_text = last_text.rstrip() + " " + chunk_text.lstrip()
                last["text"] = merged_text
                self._merge_metadata(last, chunk)
            else:
                result.append(chunk)

        return result

    def _should_merge(
        self, a: str, b: str, query_keywords: set[str]
    ) -> bool:
        if not a or not b:
            return False
        a_tokens = _tokenize(a)
        b_tokens = _tokenize(b)
        if not a_tokens or not b_tokens:
            return False

        overlap = len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))
        if overlap > 0.4:
            return True

        a_end = a[-80:].lower().strip()
        b_start = b[:80].lower().strip()
        end_start_overlap = _tokenize(a_end) & _tokenize(b_start)
        if len(end_start_overlap) >= 2:
            return True

        q_overlap_a = len(query_keywords & a_tokens)
        q_overlap_b = len(query_keywords & b_tokens)
        if q_overlap_a >= 1 and q_overlap_b >= 1 and overlap > 0.2:
            return True

        return False

    def _extract_critical(
        self, chunks: list[dict], query_keywords: set[str]
    ) -> list[dict]:
        result = []
        for chunk in chunks:
            text = _get_text(chunk)
            if len(text) <= _MIN_CHUNK_LENGTH * 2:
                result.append(chunk)
                continue

            important_sentences = self._get_important_sentences(text, query_keywords)

            if important_sentences:
                compressed = " ".join(important_sentences)
                if len(compressed) < _MIN_CHUNK_LENGTH:
                    compressed = text[:500]
                chunk["text"] = compressed
                chunk["_compressed"] = True
            result.append(chunk)

        return result

    def _get_important_sentences(
        self, text: str, query_keywords: set[str]
    ) -> list[str]:
        sentences = _SENTENCE_SPLIT_RE.split(text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

        if len(sentences) <= 3:
            return sentences

        if not query_keywords:
            return sentences[:5]

        scored = []
        for s in sentences:
            s_lower = s.lower()
            s_tokens = _tokenize(s)

            kw_count = sum(1 for kw in query_keywords if kw in s_lower)
            kw_density = kw_count / max(1, len(s_tokens))

            contains_numeric = bool(re.search(r"\d+", s))

            first_sentence_bonus = 0.15 if len(scored) == 0 else 0
            last_sentence_bonus = 0.1 if len(scored) == len(sentences) - 1 else 0

            score = kw_count * 0.3 + kw_density * 0.4 + contains_numeric * 0.15 + first_sentence_bonus + last_sentence_bonus
            scored.append((score, s))

        scored.sort(key=lambda x: -x[0])

        top_sentences = [s for _, s in scored[:5]]

        first = sentences[0]
        if first not in top_sentences:
            top_sentences.insert(0, first)

        return top_sentences[:6]

    def _enforce_budget(
        self, chunks: list[dict], max_chars: int
    ) -> list[dict]:
        if not chunks:
            return []

        dynamic_budget = self._dynamic_budget(chunks, max_chars)

        total = sum(len(_get_text(c)) for c in chunks)
        if total <= dynamic_budget:
            return chunks

        scored = []
        for c in chunks:
            score = float(
                c.get("_relevance_score", 0)
                or c.get("score", 0)
                or 0.1
            )
            text = _get_text(c)
            scored.append((score, len(text), c))

        scored.sort(key=lambda x: (-x[0], x[1]))

        result = []
        used = 0
        for score, length, chunk in scored:
            if used + length > dynamic_budget:
                remaining = dynamic_budget - used
                if remaining > 200:
                    text = _get_text(chunk)
                    chunk["text"] = text[:remaining] + "..."
                    result.append(chunk)
                break
            result.append(chunk)
            used += length

        result.sort(key=lambda x: x.get("_original_index", 0))
        return result or [chunks[0]]

    def _dynamic_budget(self, chunks: list[dict], max_chars: int) -> int:
        avg_score = 0.0
        count = 0
        for c in chunks:
            s = float(c.get("_relevance_score", 0) or c.get("score", 0) or 0)
            if s > 0:
                avg_score += s
                count += 1
        if count > 0:
            avg_score /= count

        if avg_score < 0.2:
            return min(max_chars, 2000)
        elif avg_score < 0.5:
            return min(max_chars, 3500)
        else:
            return min(max_chars, int(max_chars * 0.8))

    def format_context(self, chunks: list[dict]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            text = _get_text(chunk)
            if text:
                parts.append(f"[{i}] {text}")
        return "\n\n".join(parts)
