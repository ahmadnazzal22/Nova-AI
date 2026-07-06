import re

MIN_ANSWER_LENGTH = 10

_HALLUCATION_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bliberria\b", re.I),
    re.compile(r"متحفزات"),
    re.compile(r"\basdasd\b", re.I),
    re.compile(r"unknown term", re.I),
    re.compile(r"not real concept", re.I),
    re.compile(r"fake (concept|term|idea|technology)", re.I),
    re.compile(r"nonexistent (concept|term|idea)", re.I),
]

_UNCERTAIN_PHRASES: list[str] = [
    "maybe", "i think", "not sure", "could be",
    "i'm not sure", "i am not sure", "perhaps",
    "possibly", "might be",
]


class ResponseValidator:

    @staticmethod
    def is_valid_answer(query: str, answer: str) -> bool:
        if not answer or len(answer.strip()) < MIN_ANSWER_LENGTH:
            return False
        lower = answer.lower()
        for pattern in _HALLUCATION_PATTERNS:
            if pattern.search(lower):
                return False
        return True

    @staticmethod
    def low_confidence(answer: str) -> bool:
        lower = answer.lower()
        return any(p in lower for p in _UNCERTAIN_PHRASES)

    @staticmethod
    def build_fallback_prompt(query: str) -> str:
        return (
            f"The previous answer was low quality or incorrect.\n\n"
            f"Question: {query}\n\n"
            f"Provide a clear, correct, and factual explanation.\n"
            f"Do NOT say 'I am not certain' or 'I am not sure'.\n"
            f"If the topic is known, explain it with confidence.\n"
            f"Use only standard and known concepts."
        )
