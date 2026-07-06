import json
import re
import time
from dataclasses import dataclass, asdict
from .logger import get_logger

logger = get_logger(__name__)

DECISION_PROMPT = "Return JSON only.\n\nDecide if tools are needed.\n\nTools: web_search, kb_search, file_search, none\n\nUser: {query}"

_DECISION_CACHE_TTL = 600


@dataclass
class ToolDecision:
    use_tools: bool = False
    tool: str = "none"
    reason: str = ""
    rewritten_query: str = ""


class ToolDecisionLLM:
    def __init__(self, llm=None):
        self.llm = llm
        self._cache: dict[str, tuple[float, ToolDecision]] = {}

    def _cache_get(self, query: str) -> ToolDecision | None:
        entry = self._cache.get(query)
        if entry:
            ts, decision = entry
            if time.time() - ts < _DECISION_CACHE_TTL:
                return decision
            del self._cache[query]
        return None

    def _cache_set(self, query: str, decision: ToolDecision):
        self._cache[query] = (time.time(), decision)
        if len(self._cache) > 500:
            stale = [k for k, v in self._cache.items() if time.time() - v[0] > _DECISION_CACHE_TTL]
            for k in stale:
                del self._cache[k]

    def set_llm(self, llm):
        if llm is not self.llm:
            self.llm = llm
            self._cache.clear()

    def decide(self, query: str) -> ToolDecision:
        if not query or not query.strip():
            return ToolDecision(use_tools=False, tool="none", reason="Empty query", rewritten_query=query)

        cached = self._cache_get(query)
        if cached:
            logger.debug("Tool decision cache hit: %.60s", query)
            return cached

        if self.llm is None:
            return ToolDecision(use_tools=False, tool="none", reason="No LLM available", rewritten_query=query)

        prompt = DECISION_PROMPT.format(query=query)
        try:
            response = self.llm.invoke(prompt)
            data = self._parse_json(response)
            decision = ToolDecision(
                use_tools=bool(data.get("use_tools", False)),
                tool=str(data.get("tool", "none")),
                reason=str(data.get("reason", "")),
                rewritten_query=str(data.get("rewritten_query", query)),
            )
            self._cache_set(query, decision)
            return decision
        except Exception as e:
            logger.warning("Tool decision LLM failed: %s", e)
            return ToolDecision(use_tools=False, tool="none", reason="Fallback: LLM decision failed", rewritten_query=query)

    def _parse_json(self, text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        text = text.strip()
        return json.loads(text)
