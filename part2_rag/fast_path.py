GREETINGS = frozenset({
    "hi", "hello", "hey", "هاي", "مرحبا", "السلام عليكم",
    "thanks", "thank you", "thankyou", "bye", "goodbye",
    "ok", "okay", "yes", "no", "sure", "great", "nice",
})


def is_fast_path(query: str) -> bool:
    """Only return True for actual greetings or empty queries.
    Everything else goes through the full pipeline."""
    q = query.lower().strip()
    if not q:
        return True
    return q in GREETINGS
