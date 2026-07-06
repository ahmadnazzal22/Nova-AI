import json
import re
from .database import MemoryRepository
from .logger import get_logger

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


class MemoryManager:
    def __init__(self, session):
        self.repo = MemoryRepository(session)

    def load_context(self, user_id: int, limit: int = 5) -> str:
        memories = self.repo.get_top(user_id, limit=limit)
        if not memories:
            return ""
        lines = ["User known information:"]
        char_count = 0
        for m in memories:
            entry = f"- {m.key}: {m.value[:120]}"
            char_count += len(entry)
            if char_count > 800:
                break
            lines.append(entry)
        return "\n".join(lines)

    def extract_and_store(self, user_id: int, question: str, answer: str, llm) -> list[dict]:
        # Skip trivial exchanges
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
            if memories:
                # Filter low-importance memories
                memories = [m for m in memories if float(m.get("importance", 0)) >= 0.3]
                if memories:
                    self.repo.store_batch(user_id, memories)
                    logger.info("Stored %d memories for user %d", len(memories), user_id)
            return memories
        except Exception as e:
            logger.warning("Memory extraction failed: %s", e)
            return []
