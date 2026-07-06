import json
import re
from ..logger import get_logger

logger = get_logger(__name__)

MEMORY_EXTRACTION_PROMPT = """From this conversation turn, extract any stable user facts.
Return ONLY valid JSON:
{{"memories": [
  {{"key": "preference", "value": "...", "importance": 0.8}},
  {{"key": "fact", "value": "...", "importance": 0.6}}
]}}

Rules:
- Only extract stable facts (preferences, goals, repeated behaviors, important info)
- Use broad keys like "preference", "goal", "fact", "topic_interest"
- importance: 0.0 (trivial) to 1.0 (critical)
- If nothing to extract, return {{"memories": []}}
- Do NOT extract conversational filler, greetings, or obvious statements
- Be concise in extracted values (max 100 chars)

Question: {question}
Answer: {answer}"""


class MemoryService:
    def __init__(self):
        self._short_term: dict[int, list[dict]] = {}
        self._max_short_term = 10

    def add_to_short_term(self, user_id: int, role: str, content: str):
        if user_id not in self._short_term:
            self._short_term[user_id] = []
        self._short_term[user_id].append({"role": role, "content": content})
        if len(self._short_term[user_id]) > self._max_short_term:
            self._short_term[user_id] = self._short_term[user_id][-self._max_short_term:]

    def get_short_term_context(self, user_id: int, limit: int = 5) -> str:
        messages = self._short_term.get(user_id, [])[-limit:]
        if not messages:
            return ""
        lines = []
        for m in messages:
            prefix = "User" if m["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {m['content'][:200]}")
        return "\n".join(lines)

    def extract_memories(self, question: str, answer: str, llm) -> list[dict]:
        if len(question.strip()) < 2 or len(answer.strip()) < 3:
            return []
        prompt = MEMORY_EXTRACTION_PROMPT.format(question=question, answer=answer[:1000])
        try:
            response = llm.invoke(prompt)
            response = response.strip()
            if response.startswith("```"):
                response = re.sub(r"^```(?:json)?\s*", "", response)
                response = re.sub(r"\s*```$", "", response)
            data = json.loads(response)
            memories = data.get("memories", [])
            return [m for m in memories if float(m.get("importance", 0)) >= 0.3]
        except Exception as e:
            logger.warning("Memory extraction failed: %s", e)
            return []

    def clear_short_term(self, user_id: int):
        self._short_term.pop(user_id, None)


_memory_service: MemoryService | None = None


def get_memory_service() -> MemoryService:
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService()
    return _memory_service
