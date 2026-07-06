"""Unit tests for QueryRouter — route_question, needs_live_search, needs_deep_research."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from part2_rag.orchestrator.query_router import (  # noqa: E402
    route_question,
    needs_live_search,
    needs_deep_research,
    Route,
)


def _route(q: str) -> Route:
    return route_question(q)


# ── Live / News ──────────────────────────────────────────────────────────────


def test_news_query():
    assert _route("what is the latest news?") == "live"
    assert _route("breaking news today") == "live"
    assert _route("tell me the latest development in AI") == "live"


def test_weather_query():
    assert _route("what is the weather in London?") == "live"
    assert _route("weather forecast for tomorrow") == "live"
    assert _route("temperature in Cairo today") == "live"


def test_price_stock_query():
    assert _route("what is the price of bitcoin?") == "live"
    assert _route("bitcoin price today") == "live"
    assert _route("stock market update") == "live"
    assert _route("how much is ethereum?") == "live"


def test_sports_query():
    assert _route("what is the score of the match?") == "live"
    assert _route("latest sport highlights") == "live"
    assert _route("who won the game yesterday?") == "live"


def test_current_event_query():
    assert _route("current status of the election") == "live"
    assert _route("who just won the election?") == "live"


def test_needs_live_search():
    assert needs_live_search("latest news") is True
    assert needs_live_search("stock price") is True
    assert needs_live_search("what is Python?") is False


# ── Research / Complex ──────────────────────────────────────────────────────


def test_compare_query():
    assert _route("compare Python and JavaScript") == "research"
    assert _route("what are the differences between AI and ML?") == "research"
    assert _route("similarities between two programming languages") == "research"


def test_analysis_query():
    assert _route("analyze the impact of climate change on agriculture") == "research"
    assert _route("what is the impact of AI on healthcare?") == "research"
    assert _route("evaluate the pros and cons of remote work") == "research"


def test_explain_query():
    assert _route("explain why the sky is blue") == "research"
    assert _route("explain how a car engine works") == "research"
    assert _route("how does the internet work?") == "research"


def test_detailed_query():
    assert _route("what are the key factors in economic growth?") == "research"
    assert _route("comprehensive analysis of renewable energy") == "research"
    assert _route("in-depth explanation of quantum computing") == "research"


def test_shorter_complex_query_not_research():
    """Short queries (<3 words) should not trigger research even if they match complex words."""
    assert _route("compare") == "rag"  # single word
    assert _route("compare things") == "rag"  # 2 words


def test_step_by_step():
    assert _route("step by step guide to baking bread") == "research"
    assert _route("what is the process of photosynthesis?") == "research"


def test_needs_deep_research():
    assert needs_deep_research("compare Python and JavaScript") is True
    assert needs_deep_research("hello") is False


# ── RAG (default) ───────────────────────────────────────────────────────────


def test_simple_rag_query():
    assert _route("what is Python?") == "rag"
    assert _route("tell me about machine learning") == "rag"
    assert _route("how do I learn programming?") == "rag"


def test_greeting_not_routed():
    """Greetings are handled by is_fast_path upstream; route_question returns rag."""
    assert _route("hello") == "rag"
    assert _route("hi") == "rag"
    assert _route("good morning") == "rag"


def test_factual_query():
    assert _route("who invented the telephone?") == "rag"
    assert _route("what is the capital of France?") == "rag"
    assert _route("when was the Eiffel Tower built?") == "rag"


def test_empty_query():
    assert _route("") == "rag"
    assert _route("   ") == "rag"


# ── Edge cases ──────────────────────────────────────────────────────────────


def test_case_insensitive_live():
    assert _route("BREAKING NEWS") == "live"
    assert _route("Latest Stock Price") == "live"
    assert _route("What Is The News Today?") == "live"


def test_mixed_keywords_live_wins():
    """Live patterns take priority over complex patterns."""
    assert _route("analyze the latest news on AI") == "live"  # 'latest news' → live


def test_special_characters():
    assert _route("what's the weather like?") == "live"
    assert _route("bitcoin price — update?") == "live"
    assert _route("hello world!") == "rag"


def test_long_text():
    long_q = " ".join(["compare"] * 10)
    assert _route(long_q) == "research"


def test_question_with_url():
    """URLs should not break routing."""
    assert _route("check out https://example.com for latest news") == "live"
    assert _route("what is the price of https://some.site?") == "live"
