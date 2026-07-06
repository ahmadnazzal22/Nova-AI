import re
from typing import Any


class CitationInjector:
    CITATION_PATTERN = re.compile(r'\[(\d+)\]')

    @staticmethod
    def build_context_with_citations(sources: list[dict]) -> tuple[str, list[dict]]:
        """Build a context string with [N] markers and return the structured citation list.

        Each source dict may have: text, title, url, source, snippet, relevance_score, etc.
        Returns (context_string, citation_list).
        """
        citation_list = []
        parts = []
        for i, s in enumerate(sources, 1):
            text = s.get("text", s.get("snippet", s.get("content", "")))
            title = s.get("title", "") or "Untitled"
            source_type = s.get("source_type", s.get("source", "kb"))
            parts.append(f"[{i}] From {title} ({source_type}): {text[:800]}")
            citation_list.append({
                "id": i,
                "text": text[:500],
                "title": title,
                "url": s.get("url", ""),
                "source": source_type,
                "relevance_score": round(s.get("relevance_score", s.get("_relevance_score", 0)), 4),
                "confidence_score": round(s.get("confidence_score", s.get("_confidence_score", 0)), 4),
            })
        context = "\n\n".join(parts)
        return context, citation_list

    @staticmethod
    def extract_citations(answer: str) -> list[int]:
        """Extract cited [N] markers from the answer text."""
        return sorted(set(int(m) for m in CitationInjector.CITATION_PATTERN.findall(answer)))

    @staticmethod
    def filter_used_citations(citation_list: list[dict], cited_ids: list[int]) -> list[dict]:
        """Return only the citations actually referenced in the answer."""
        cited_set = set(cited_ids)
        return [c for c in citation_list if c["id"] in cited_set]

    @staticmethod
    def build_source_section(citation_list: list[dict]) -> str:
        """Build a __SOURCES__ appendix string from citations."""
        if not citation_list:
            return ""
        lines = ["\n\n__SOURCES__"]
        for c in citation_list:
            lines.append(f"[{c['id']}] {c['title']} ({c['source']}, score={c['relevance_score']})")
        return "\n".join(lines)
