from .classifier import classify_query, ClassifierResult
from ..llm.llm_service import get_llm_service
from ..logger import get_logger

logger = get_logger(__name__)


def get_prompt_for_intent(intent: str, question: str, context: str = "") -> str:
    templates = {
        "GREETING": "You are a helpful assistant. Respond warmly and briefly.\n\nUser: {question}\nAssistant:",
        "HELP": "You are a helpful assistant. Ask what the user needs help with.\n\nUser: {question}\nAssistant:",
        "SPORTS": "عذراً، أنا متخصص بالذكاء الاصطناعي فقط ولا أملك بيانات رياضية. ابحث في Google أو تطبيق ESPN.",
        "SIMPLE_LLM": "You are a professional AI assistant. Answer concisely and accurately.\n\nQuestion: {question}\nAnswer:",
        "CODE": "You are an expert programmer. Provide clean, efficient code with brief explanation.\n\nRequest: {question}\nResponse:",
        "COMPLEX_RAG": "You are a professional AI assistant. Answer using the provided context when relevant.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:",
        "LIVE_SEARCH": "You are a professional AI assistant with access to live web data.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:",
    }
    t = templates.get(intent, templates["SIMPLE_LLM"])
    if intent in ("SPORTS",):
        return t
    return t.format(question=question, context=context)


def route_query(question: str, context: str = "") -> tuple[str, str]:
    result = classify_query(question)
    logger.debug("Query routed: %s (conf=%.2f, reason=%s)", result.intent, result.confidence, result.reason)
    prompt = get_prompt_for_intent(result.intent, question, context)
    return prompt, result.intent
