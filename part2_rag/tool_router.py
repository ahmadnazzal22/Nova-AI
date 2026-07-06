import re
from .logger import get_logger

logger = get_logger(__name__)

_TOOL_TRIGGERS: list[re.Pattern] = [
    re.compile(r"(latest|breaking|current|recent|today|this week|this month)\s+(news|updates|events?|happenings?)", re.I),
    re.compile(r"(what's new|anything new|what else is)", re.I),
    re.compile(r"(weather|temperature|forecast|humidity|rain|snow)\s+(today|now|currently|this)", re.I),
    re.compile(r"(stock|price|market|crypto|bitcoin|ethereum)\s+(price|value|rate|today)", re.I),
    re.compile(r"(schedule|show|movie|event|concert)\s+(today|tonight|this week)", re.I),
    re.compile(r"(search|look up|find|retrieve|fetch|google)\s", re.I),
    re.compile(r"(show|provide|give|cite|list)\s+(sources|references|citations|links)", re.I),
    re.compile(r"(in|from|according to)\s+(the|my|this)\s+(file|document|pdf|upload|paper|report)", re.I),
    re.compile(r"what does the (file|document|paper|report) say", re.I),
    re.compile(r"i uploaded|i attached|i attached|the file I", re.I),
]


def needs_tools(query: str) -> bool:
    if not query or not query.strip():
        return False
    q = query.strip().lower()
    for pattern in _TOOL_TRIGGERS:
        if pattern.search(q):
            return True
    return False
