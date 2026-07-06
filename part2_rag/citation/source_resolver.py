import re
from typing import Any


class SourceResolver:
    CITATION_PATTERN = re.compile(r'\[(\d+)\]')

    @staticmethod
    def parse_answer_citations(answer: str) -> list[int]:
        """Extract all [N] citation IDs from the answer text in order of appearance."""
        return [int(m) for m in SourceResolver.CITATION_PATTERN.findall(answer)]

    @staticmethod
    def resolve_citations(answer: str, citations: list[dict]) -> dict[str, Any]:
        """Given an answer and a full citation list, return only the cited entries.

        Returns: {citations: [cited CitationItems], uncited_count: int}
        """
        cited_ids = set(SourceResolver.parse_answer_citations(answer))
        used = [c for c in citations if c["id"] in cited_ids]
        return {"citations": used, "uncited_count": len(citations) - len(used)}

    @staticmethod
    def render_answer_with_sources(answer: str, citations: list[dict]) -> tuple[str, str]:
        """Append a sources appendix to the answer based on cited [N] markers.

        Returns (answer_with_sources, sources_text).
        """
        if not citations:
            return answer, ""
        parts = ["\n\n**Sources:**"]
        for c in citations:
            parts.append(f"- [{c['id']}] {c['title']} ({c['source']}) — score {c['relevance_score']}")
        sources_text = "\n".join(parts)
        return answer + sources_text, sources_text
