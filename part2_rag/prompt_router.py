from .response_formatter import ResponseFormatter
from .prompt_templates import INTENT_PROMPT_MAP, CHAT_PROMPT, CODE_PROMPT

GREETING_KEYWORDS = frozenset({
    "hi", "hello", "hey", "هاي", "مرحبا", "السلام عليكم",
    "thanks", "thank you", "thankyou", "bye", "goodbye",
    "ok", "okay", "yes", "no", "sure", "great", "nice",
    "good morning", "good evening", "good afternoon",
})

CODING_STARTS = (
    "write code", "write a", "write an",
    "python", "javascript", "typescript", "java", "rust",
    "go lang", "coding", "debug", "debugging",
    "function", "class", "implement",
)

_formatter = ResponseFormatter()


def _is_greeting(query: str) -> bool:
    return query.lower().strip() in GREETING_KEYWORDS


def _is_coding(query: str) -> bool:
    q = query.lower().strip()
    if q.startswith(CODING_STARTS):
        return True
    words = set(q.split())
    code_keywords = {"python", "javascript", "typescript", "java", "rust", "go", "code", "coding", "debug", "debugging", "function", "class", "algorithm", "api"}
    if words & code_keywords:
        return True
    return False


def select_prompt(question: str) -> str:
    if _is_greeting(question):
        return CHAT_PROMPT
    if _is_coding(question):
        return CODE_PROMPT
    # Use ResponseFormatter's intent detection for the rest
    intent = _formatter._detect_intent(question)
    return INTENT_PROMPT_MAP.get(intent, INTENT_PROMPT_MAP["general"])
