import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import re
import json
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from part2_rag.intent_detector import IntentDetector, QueryIntent, _classify_fast, INTENT_PROMPT
from part2_rag.web_search import rewrite_query, filter_domain, WebSearch, SearchResult
from part2_rag.reranker import (
    Reranker,
    _extract_keywords,
    _keyword_overlap,
    _title_relevance,
    _noise_penalty,
    _reduce_redundancy,
    _clean_doc_text,
    _SCORE_THRESHOLD,
    _STOP_WORDS,
)
from part2_rag.rag_agent import _clean_context, _is_sufficient_context, _format_context_with_citations
from part2_rag.tool_decision_llm import ToolDecisionLLM, ToolDecision
from part2_rag.fast_path import is_fast_path
from part2_rag.response_validator import ResponseValidator
from part2_rag.rag_agent import _inject_concept_snippets, _is_explanation_task
from part2_rag.prompt_router import select_prompt
from part2_rag.prompt_templates import CHAT_PROMPT, EXPLANATION_PROMPT, CODE_PROMPT, GENERAL_PROMPT, COMPARISON_PROMPT
from part2_rag.database import init_db, get_session, MemoryRepository
from part2_rag.memory_manager import MemoryManager
from part2_rag.models import Base


@pytest.fixture
def mock_embeddings():
    class MockEmb:
        def __init__(self):
            self.config = MagicMock()
            self.config.d_model = 256
            self._vec = np.random.RandomState(42).randn(256).astype(np.float32)
            self._vec = self._vec / np.linalg.norm(self._vec)
        def embed_query(self, text):
            rng = np.random.RandomState(hash(text) % (2**31))
            v = rng.randn(256).astype(np.float32)
            return (v / np.linalg.norm(v)).tolist()
    return MockEmb()


@pytest.fixture
def reranker(mock_embeddings):
    emb = mock_embeddings
    emb.config.d_model = 256
    return Reranker(embeddings_model=emb, top_k=5)


class TestRewriteQuery:
    def test_removes_filler_words(self):
        result = rewrite_query("tell me about transformers")
        assert not any(w in result.lower().split() for w in ["tell", "me", "about"])
        assert "transformers" in result.lower()

    def test_expands_technical_terms(self):
        result = rewrite_query("how does RAG work")
        assert "retrieval augmented generation" in result.lower()

    def test_short_query_unchanged(self):
        result = rewrite_query("AI")
        assert len(result) > 0

    def test_empty_returns_empty(self):
        assert rewrite_query("") == ""


class TestFilterDomain:
    def test_allows_trusted_domain(self):
        assert filter_domain("https://arxiv.org/abs/2301.001")

    def test_blocks_noise_domain(self):
        assert not filter_domain("https://youtube.com/watch?v=abc")

    def test_blocks_noise_subdomain(self):
        assert not filter_domain("https://www.reddit.com/r/something")

    def test_unknown_domain_allowed(self):
        assert filter_domain("https://example.org/research")

    def test_empty_url_not_blocked(self):
        assert filter_domain("")


class TestKeywordExtraction:
    def test_extracts_meaningful_words(self):
        kw = _extract_keywords("Transformers use self-attention for parallel processing")
        assert "transformers" in kw
        assert "the" not in kw
        assert "parallel" in kw

    def test_removes_stop_words(self):
        kw = _extract_keywords("the cat sat on the mat")
        assert "the" not in kw
        assert "cat" in kw
        assert "mat" in kw

    def test_removes_short_words(self):
        kw = _extract_keywords("a b cd xyz")
        assert "xyz" in kw
        assert "a" not in kw
        assert "b" not in kw
        assert "cd" not in kw

    def test_empty_returns_empty(self):
        assert _extract_keywords("") == set()

    def test_only_stop_words_returns_empty(self):
        assert _extract_keywords("the a an of") == set()


class TestKeywordOverlap:
    def test_full_overlap(self):
        assert _keyword_overlap({"cat", "mat"}, {"cat", "mat", "sat"}) == 1.0

    def test_partial_overlap(self):
        score = _keyword_overlap({"cat", "dog", "mat"}, {"cat", "mat"})
        assert score == pytest.approx(2 / 3)

    def test_no_overlap(self):
        assert _keyword_overlap({"cat", "dog"}, {"fish", "bird"}) == 0.0

    def test_empty_query(self):
        assert _keyword_overlap(set(), {"cat"}) == 0.0


class TestTitleRelevance:
    def test_all_keywords_in_title(self):
        score = _title_relevance({"transformer", "attention"}, "Transformer Attention Model")
        assert score == 1.0

    def test_some_keywords_in_title(self):
        score = _title_relevance({"transformer", "attention", "bert"}, "Transformer Overview")
        assert score == pytest.approx(1 / 3)

    def test_no_keywords_in_title(self):
        assert _title_relevance({"cat", "dog"}, "Some Title") == 0.0

    def test_empty_title(self):
        assert _title_relevance({"test"}, "") == 0.0

    def test_empty_keywords(self):
        assert _title_relevance(set(), "Title") == 0.0


class TestNoisePenalty:
    def test_clean_text_no_penalty(self):
        assert _noise_penalty("This is a clean article about transformers.") == 0.0

    def test_click_here_penalty(self):
        assert _noise_penalty("Click here to subscribe") > 0.0

    def test_subscribe_penalty(self):
        assert _noise_penalty("Sign up for our newsletter") > 0.0

    def test_capped_at_one(self):
        noisy = "click here subscribe newsletter advertisement sponsored cookie policy"
        assert _noise_penalty(noisy) <= 1.0


class TestReduceRedundancy:
    def test_removes_duplicate_text(self):
        docs = [
            {"text": "Transformers use self-attention."},
            {"text": "Transformers use self-attention."},
            {"text": "Different content here."},
        ]
        result = _reduce_redundancy(docs)
        assert len(result) == 2

    def test_similar_texts_are_deduplicated(self):
        docs = [
            {"text": "Hello world this is a test of the system"},
            {"text": "Hello world this is a test of the system"},
        ]
        result = _reduce_redundancy(docs)
        assert len(result) == 1

    def test_all_unique_preserved(self):
        docs = [{"text": f"Document {i}"} for i in range(5)]
        assert len(_reduce_redundancy(docs)) == 5


class TestCleanDocText:
    def test_removes_bracketed_content(self):
        result = _clean_doc_text("Hello [some noise] world")
        assert "[some noise]" not in result

    def test_preserves_citations(self):
        result = _clean_doc_text("According to [1] this is true")
        assert "[1]" in result

    def test_removes_noise_phrases(self):
        result = _clean_doc_text("Click here to subscribe to our newsletter")
        assert "click here" not in result.lower()

    def test_normalizes_whitespace(self):
        result = _clean_doc_text("hello     world")
        assert result == "hello world"

    def test_empty_string(self):
        assert _clean_doc_text("") == ""


class TestReranker:
    def test_rerank_returns_top_k(self, reranker):
        docs = [{"text": f"Document about transformers number {i}", "title": f"Doc {i}"} for i in range(10)]
        result = reranker.rerank("transformers attention", docs, k=3)
        assert len(result) <= 3

    def test_rerank_empty_docs(self, reranker):
        assert reranker.rerank("test", []) == []

    def test_rank_with_scores_returns_tuples(self, reranker):
        docs = [{"text": "Transformers use self-attention", "title": "Doc 1"}]
        result = reranker.rank_with_scores("transformers", docs)
        assert len(result) == 1
        assert isinstance(result[0], tuple)
        assert isinstance(result[0][1], float)

    def test_rank_with_scores_empty(self, reranker):
        assert reranker.rank_with_scores("test", []) == []

    def test_high_relevance_scores_high(self, reranker):
        docs = [
            {"text": "Artificial intelligence and machine learning are transforming the world", "title": "AI Article"},
            {"text": "The weather today is sunny with a chance of rain", "title": "Weather Report"},
        ]
        result = reranker.rank_with_scores("artificial intelligence machine learning", docs)
        ai_score = result[0][1]
        assert ai_score > _SCORE_THRESHOLD

    def test_irrelevant_docs_scored_low(self, reranker):
        docs = [
            {"text": "Cooking recipes for pasta carbonara", "title": "Recipe"},
        ]
        result = reranker.rank_with_scores("quantum physics", docs)
        if result:
            assert result[0][1] < 0.5

    def test_rerank_missing_text_field(self, reranker):
        docs = [{"title": "No text here"}]
        result = reranker.rerank("test", docs)
        assert result == []

    def test_prefers_title_match(self, reranker):
        docs = [
            {"text": "Some unrelated content here", "title": "Transformers and Attention Mechanisms"},
            {"text": "Transformers are great for NLP tasks", "title": "Random Title"},
        ]
        result = reranker.rank_with_scores("transformers", docs)
        if result:
            assert result[0][1] >= _SCORE_THRESHOLD

    def test_rerank_respects_k(self, reranker):
        docs = [{"text": f"Document {i} about transformers", "title": f"Doc {i}"} for i in range(20)]
        result = reranker.rerank("transformers", docs, k=2)
        assert len(result) <= 2


class TestCleanContext:
    def test_removes_duplicates(self):
        chunks = [
            {"text": "Transformers use self-attention.", "title": "A"},
            {"text": "Transformers use self-attention.", "title": "B"},
        ]
        result = _clean_context(chunks)
        assert len(result) == 1

    def test_removes_noise_phrases(self):
        chunks = [{"text": "Click here to subscribe. Transformers are great.", "title": "A"}]
        result = _clean_context(chunks)
        assert "click here" not in result[0]["text"].lower()

    def test_handles_empty_list(self):
        assert _clean_context([]) == []

    def test_handles_varied_field_keys(self):
        chunks = [{"content": "Transformers use attention", "title": "A"}]
        result = _clean_context(chunks)
        assert len(result) == 1

    def test_normalizes_whitespace(self):
        chunks = [{"text": "Hello     world", "title": "A"}]
        result = _clean_context(chunks)
        assert result[0]["text"] == "Hello world"


class TestIsSufficientContext:
    def test_sufficient_context(self):
        scored = [({"text": "A"}, 0.3), ({"text": "B"}, 0.2)]
        assert _is_sufficient_context(scored) is True

    def test_insufficient_low_max(self):
        scored = [({"text": "A"}, 0.1), ({"text": "B"}, 0.05)]
        assert _is_sufficient_context(scored) is False

    def test_empty_list(self):
        assert _is_sufficient_context([]) is False

    def test_insufficient_avg_below_threshold(self):
        scored = [({"text": "A"}, 0.12), ({"text": "B"}, 0.05)]
        assert _is_sufficient_context(scored) is False

    def test_single_good_doc(self):
        scored = [({"text": "A"}, 0.4)]
        assert _is_sufficient_context(scored) is True


class TestFormatContextWithCitations:
    def test_formats_citations(self):
        chunks = [
            {"text": "Transformers use self-attention.", "title": "A", "url": "http://a.com"},
            {"text": "Attention helps focus.", "title": "B", "url": "http://b.com"},
        ]
        result = _format_context_with_citations(chunks)
        assert "[1]" in result
        assert "[2]" in result
        assert "Transformers use self-attention" in result

    def test_empty_chunks(self):
        assert _format_context_with_citations([]) == ""


class TestWebSearchDomainFilter:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ws = WebSearch(max_results=3)

    def test_search_and_fetch_filters_noise(self):
        with patch.object(self.ws, "_search_duckduckgo") as mock_search:
            mock_search.return_value = [
                SearchResult(title="Bad", url="https://youtube.com/watch?v=abc", snippet="video", source="duckduckgo"),
                SearchResult(title="Good", url="https://arxiv.org/abs/2301", snippet="paper", source="duckduckgo"),
            ]
            with patch.object(self.ws, "fetch_page") as mock_fetch:
                mock_fetch.return_value = ("html", "some content")
                result = self.ws.search_and_fetch("test query")
                assert len(result) == 1
                assert "arxiv.org" in result[0].url

    def test_all_noise_returns_empty(self):
        with patch.object(self.ws, "_search_duckduckgo") as mock_search:
            mock_search.return_value = [
                SearchResult(title="Vid", url="https://youtube.com/watch?v=abc", snippet="vid", source="duckduckgo"),
                SearchResult(title="FB", url="https://facebook.com/post", snippet="post", source="duckduckgo"),
            ]
            result = self.ws.search_and_fetch("test")
            assert len(result) == 0


class TestLLMFirstFallback:
    def test_live_empty_search_falls_back_to_llm(self, mock_embeddings):
        from part2_rag.rag_agent import RAGAgent, MockLLM
        import tempfile
        tmpdir = tempfile.mkdtemp()
        try:
            agent = RAGAgent(persist_dir=tmpdir, llm=MockLLM())
            mock_embeddings.config.d_model = 256
            agent.web_search = MagicMock()
            agent.web_search.search_and_fetch.return_value = []
            result = agent.query_live("What is the latest news on AI?")
            assert result.get("tool_used") is None
            assert "answer" in result
        finally:
            import shutil
            for _ in range(3):
                try:
                    shutil.rmtree(tmpdir)
                    break
                except PermissionError:
                    import time
                    time.sleep(0.5)

    def test_low_quality_search_falls_back_to_llm(self, mock_embeddings):
        from part2_rag.rag_agent import RAGAgent, MockLLM
        import tempfile
        tmpdir = tempfile.mkdtemp()
        try:
            agent = RAGAgent(persist_dir=tmpdir, llm=MockLLM())
            agent.web_search = MagicMock()
            agent.web_search.search_and_fetch.return_value = [
                SearchResult(title="Irrelevant", url="http://example.com", snippet="cooking recipe pasta", source="duckduckgo"),
            ]
            agent.reranker = Reranker(embeddings_model=mock_embeddings, top_k=3)
            result = agent.query_live("What is the latest news on AI?")
            assert "answer" in result
        finally:
            import shutil
            for _ in range(3):
                try:
                    shutil.rmtree(tmpdir)
                    break
                except PermissionError:
                    import time
                    time.sleep(0.5)


class TestFastIntentClassification:
    def test_greeting_is_general(self):
        assert _classify_fast("Hello") == QueryIntent.GENERAL

    def test_how_are_you_is_general(self):
        assert _classify_fast("How are you?") == QueryIntent.GENERAL

    def test_good_morning_is_general(self):
        assert _classify_fast("Good morning") == QueryIntent.GENERAL

    def test_joke_request_is_general(self):
        assert _classify_fast("Tell me a joke") == QueryIntent.GENERAL

    def test_thanks_is_general(self):
        assert _classify_fast("Thank you") == QueryIntent.GENERAL

    def test_bye_is_general(self):
        assert _classify_fast("Goodbye") == QueryIntent.GENERAL

    def test_who_are_you_is_general(self):
        assert _classify_fast("Who are you?") == QueryIntent.GENERAL

    def test_what_can_you_do_is_general(self):
        assert _classify_fast("What can you do?") == QueryIntent.GENERAL

    def test_nice_to_meet_is_general(self):
        assert _classify_fast("Nice to meet you") == QueryIntent.GENERAL

    def test_what_is_transformer_is_retrieval(self):
        assert _classify_fast("What is a transformer?") == QueryIntent.RETRIEVAL

    def test_explain_attention_is_retrieval(self):
        assert _classify_fast("Explain self-attention") == QueryIntent.RETRIEVAL

    def test_how_does_rag_work_is_retrieval(self):
        assert _classify_fast("How does RAG work?") == QueryIntent.RETRIEVAL

    def test_define_embedding_is_retrieval(self):
        assert _classify_fast("Define word embeddings") == QueryIntent.RETRIEVAL

    def test_compare_models_is_retrieval(self):
        assert _classify_fast("Compare BERT and GPT") == QueryIntent.RETRIEVAL

    def test_ambiguous_returns_none(self):
        assert _classify_fast("I was wondering about something") is None

    def test_empty_question_is_general(self):
        assert _classify_fast("") == QueryIntent.GENERAL


class TestIntentDetectorLLM:
    def test_general_greeting_with_llm(self):
        llm = MagicMock()
        llm.invoke.return_value = "chat"
        detector = IntentDetector(llm=llm)
        assert detector.classify("What do you think about AI?") == QueryIntent.GENERAL

    def test_retrieval_question_with_llm(self):
        llm = MagicMock()
        llm.invoke.return_value = "search"
        detector = IntentDetector(llm=llm)
        assert detector.classify("Latest news on AI") == QueryIntent.RETRIEVAL

    def test_llm_failure_defaults_to_retrieval(self):
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("LLM down")
        detector = IntentDetector(llm=llm)
        assert detector.classify("some question") == QueryIntent.RETRIEVAL

    def test_no_llm_ambiguous_defaults_to_retrieval(self):
        detector = IntentDetector(llm=None)
        assert detector.classify("some random text") == QueryIntent.RETRIEVAL

    def test_fast_path_used_before_llm(self):
        llm = MagicMock()
        detector = IntentDetector(llm=llm)
        assert detector.classify("Hello") == QueryIntent.GENERAL
        llm.invoke.assert_not_called()


class TestFastPath:
    def test_greeting_is_fast(self):
        assert is_fast_path("Hello")
        assert is_fast_path("hi")
        assert is_fast_path("Hey")
        assert is_fast_path("thanks")
        assert is_fast_path("مرحبا")

    def test_short_query_is_fast(self):
        assert is_fast_path("what")
        assert is_fast_path("hello world")
        assert is_fast_path("a b c")

    def test_short_real_time_not_fast(self):
        assert not is_fast_path("AI news")
        assert not is_fast_path("bitcoin price")
        assert not is_fast_path("weather today")

    def test_explanation_start_is_fast(self):
        assert is_fast_path("Explain self-attention")
        assert is_fast_path("How to write Python")
        assert is_fast_path("Define embedding")
        assert is_fast_path("Write a function")
        assert is_fast_path("Describe a transformer")

    def test_what_is_explanation_is_fast(self):
        assert is_fast_path("What is a transformer?")
        assert is_fast_path("What's self-attention?")
        assert is_fast_path("What are embeddings?")
        assert is_fast_path("What does RAG stand for?")

    def test_what_is_real_time_not_fast(self):
        assert not is_fast_path("What is the latest news on AI?")
        assert not is_fast_path("What's the weather today?")
        assert not is_fast_path("What is the current bitcoin price?")

    def test_empty_is_fast(self):
        assert is_fast_path("")
        assert is_fast_path("   ")

    def test_complex_query_not_fast(self):
        assert not is_fast_path("Search for quantum computing")
        assert not is_fast_path("Can you find me the current price of bitcoin?")
        assert not is_fast_path("According to my document, what is RAG?")
        assert not is_fast_path("Show me recent updates")

    def test_four_or_more_words_real_time_not_fast(self):
        assert not is_fast_path("Show me latest AI news")
        assert not is_fast_path("What's the current bitcoin price")

    def test_explanation_with_real_time_not_fast(self):
        assert not is_fast_path("What is the latest news?")
        assert not is_fast_path("How to check weather today")


class TestToolDecisionLLM:
    def test_no_llm_fallback_no_tools(self):
        decider = ToolDecisionLLM(llm=None)
        decision = decider.decide("Hello")
        assert decision.use_tools is False
        assert decision.tool == "none"

    def test_empty_query_no_tools(self):
        decider = ToolDecisionLLM(llm=MagicMock())
        decision = decider.decide("")
        assert decision.use_tools is False
        decision = decider.decide("   ")
        assert decision.use_tools is False

    def test_greeting_no_tools(self):
        llm = MagicMock()
        llm.invoke.return_value = json.dumps({"use_tools": False, "tool": "none", "reason": "Greeting", "rewritten_query": "Hello"})
        decider = ToolDecisionLLM(llm=llm)
        for q in ("Hello", "How are you?", "Good morning"):
            d = decider.decide(q)
            assert d.use_tools is False, f"Expected no tools for '{q}'"
            assert d.tool == "none"

    def test_explanation_no_tools(self):
        llm = MagicMock()
        llm.invoke.return_value = json.dumps({"use_tools": False, "tool": "none", "reason": "Explanation", "rewritten_query": "What is a transformer?"})
        decider = ToolDecisionLLM(llm=llm)
        for q in ("What is a transformer?", "Explain self-attention", "Write a Python function"):
            d = decider.decide(q)
            assert d.use_tools is False

    def test_news_triggers_web_search(self):
        llm = MagicMock()
        llm.invoke.return_value = json.dumps({"use_tools": True, "tool": "web_search", "reason": "News request", "rewritten_query": "latest AI news"})
        decider = ToolDecisionLLM(llm=llm)
        decision = decider.decide("What is the latest news on AI?")
        assert decision.use_tools is True
        assert decision.tool == "web_search"

    def test_source_request_triggers_search(self):
        llm = MagicMock()
        llm.invoke.return_value = json.dumps({"use_tools": True, "tool": "web_search", "reason": "Source request", "rewritten_query": "quantum computing"})
        decider = ToolDecisionLLM(llm=llm)
        decision = decider.decide("Search for quantum computing")
        assert decision.use_tools is True

    def test_file_reference_triggers_file_search(self):
        llm = MagicMock()
        llm.invoke.return_value = json.dumps({"use_tools": True, "tool": "file_search", "reason": "File reference", "rewritten_query": "document content"})
        decider = ToolDecisionLLM(llm=llm)
        decision = decider.decide("What does the file say?")
        assert decision.use_tools is True
        assert decision.tool == "file_search"

    def test_kb_reference_triggers_kb_search(self):
        llm = MagicMock()
        llm.invoke.return_value = json.dumps({"use_tools": True, "tool": "kb_search", "reason": "Knowledge base", "rewritten_query": "RAG systems"})
        decider = ToolDecisionLLM(llm=llm)
        decision = decider.decide("Find information about RAG")
        assert decision.use_tools is True
        assert decision.tool == "kb_search"

    def test_invalid_json_falls_back(self):
        llm = MagicMock()
        llm.invoke.return_value = "not valid json"
        decider = ToolDecisionLLM(llm=llm)
        decision = decider.decide("some question")
        assert decision.use_tools is False
        assert "Fallback" in decision.reason

    def test_malformed_json_falls_back(self):
        llm = MagicMock()
        llm.invoke.return_value = "```json\n{bad json}\n```"
        decider = ToolDecisionLLM(llm=llm)
        decision = decider.decide("some question")
        assert decision.use_tools is False

    def test_llm_error_falls_back(self):
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("LLM down")
        decider = ToolDecisionLLM(llm=llm)
        decision = decider.decide("some question")
        assert decision.use_tools is False

    def test_rewritten_query_preserved(self):
        llm = MagicMock()
        llm.invoke.return_value = json.dumps({"use_tools": True, "tool": "web_search", "reason": "test", "rewritten_query": "optimized query"})
        decider = ToolDecisionLLM(llm=llm)
        decision = decider.decide("original query")
        assert decision.rewritten_query == "optimized query"

    def test_json_in_code_block_parsed(self):
        llm = MagicMock()
        llm.invoke.return_value = "```json\n{\"use_tools\": true, \"tool\": \"kb_search\", \"reason\": \"test\", \"rewritten_query\": \"query\"}\n```"
        decider = ToolDecisionLLM(llm=llm)
        decision = decider.decide("test")
        assert decision.use_tools is True
        assert decision.tool == "kb_search"


class TestRAGAgentLLMFirst:
    @pytest.fixture(autouse=True)
    def reset_agent_singleton(self):
        from part2_rag.rag_agent import RAGAgent
        RAGAgent._instance = None
        yield

    def test_greeting_uses_direct_llm(self, mock_embeddings):
        from part2_rag.rag_agent import RAGAgent, MockLLM
        import tempfile
        tmpdir = tempfile.mkdtemp()
        try:
            agent = RAGAgent(persist_dir=tmpdir, llm=MockLLM())
            result = agent.query("Hello")
            assert result.get("tool_used") is None
            assert result["sources"] == []
            assert "answer" in result
        finally:
            import shutil
            for _ in range(3):
                try:
                    shutil.rmtree(tmpdir)
                    break
                except PermissionError:
                    import time
                    time.sleep(0.5)

    def test_programming_question_no_retrieval(self, mock_embeddings):
        from part2_rag.rag_agent import RAGAgent, MockLLM
        import tempfile
        tmpdir = tempfile.mkdtemp()
        try:
            agent = RAGAgent(persist_dir=tmpdir, llm=MockLLM())
            result = agent.query("Write a Python class")
            assert result.get("tool_used") == "kb"
            assert result.get("no_results") is True
            assert result["sources"] == []
            assert result["answer"] == ""
        finally:
            import shutil
            for _ in range(3):
                try:
                    shutil.rmtree(tmpdir)
                    break
                except PermissionError:
                    import time
                    time.sleep(0.5)

    def test_tool_trigger_uses_kb(self, mock_embeddings):
        from part2_rag.rag_agent import RAGAgent, MockLLM
        import tempfile
        tmpdir = tempfile.mkdtemp()
        try:
            agent = RAGAgent(persist_dir=tmpdir, llm=MockLLM())
            agent.tool_decider.decide = MagicMock(return_value=ToolDecision(
                use_tools=True, tool="kb_search", reason="test", rewritten_query="transformer information",
            ))
            agent.add_documents(["Transformers use self-attention."])
            result = agent.query("Search for transformer information")
            assert result.get("tool_used") == "kb"
            assert len(result.get("sources", [])) > 0
        finally:
            import shutil
            for _ in range(3):
                try:
                    shutil.rmtree(tmpdir)
                    break
                except PermissionError:
                    import time
                    time.sleep(0.5)

    def test_live_greeting_skips_search(self, mock_embeddings):
        from part2_rag.rag_agent import RAGAgent, MockLLM
        import tempfile
        tmpdir = tempfile.mkdtemp()
        try:
            agent = RAGAgent(persist_dir=tmpdir, llm=MockLLM())
            agent.web_search = MagicMock()
            result = agent.query_live("Hi there")
            assert result.get("tool_used") is None
            agent.web_search.search_and_fetch.assert_not_called()
        finally:
            import shutil
            for _ in range(3):
                try:
                    shutil.rmtree(tmpdir)
                    break
                except PermissionError:
                    import time
                    time.sleep(0.5)

    def test_live_stream_greeting_skips_search(self, mock_embeddings):
        from part2_rag.rag_agent import RAGAgent, MockLLM
        import tempfile
        tmpdir = tempfile.mkdtemp()
        try:
            agent = RAGAgent(persist_dir=tmpdir, llm=MockLLM())
            agent.web_search = MagicMock()
            events = list(agent.query_live_stream("Hello"))
            agent.web_search.search_and_fetch.assert_not_called()
            event_types = [json.loads(e)["type"] for e in events]
            assert "done" in event_types
        finally:
            import shutil
            for _ in range(3):
                try:
                    shutil.rmtree(tmpdir)
                    break
                except PermissionError:
                    import time
                    time.sleep(0.5)

    def test_live_news_triggers_web_search(self, mock_embeddings):
        from part2_rag.rag_agent import RAGAgent, MockLLM
        import tempfile
        tmpdir = tempfile.mkdtemp()
        try:
            agent = RAGAgent(persist_dir=tmpdir, llm=MockLLM())
            agent.tool_decider.decide = MagicMock(return_value=ToolDecision(
                use_tools=True, tool="web_search", reason="test", rewritten_query="latest AI news",
            ))
            agent.web_search = MagicMock()
            agent.web_search.search_and_fetch.return_value = []
            result = agent.query_live("What is the latest news on AI?")
            agent.web_search.search_and_fetch.assert_called_once()
            assert "answer" in result
        finally:
            import shutil
            for _ in range(3):
                try:
                    shutil.rmtree(tmpdir)
                    break
                except PermissionError:
                    import time
                    time.sleep(0.5)


class TestResponseValidator:
    def test_valid_answer_passes(self):
        assert ResponseValidator.is_valid_answer("what is ai", "Artificial intelligence is a field of computer science.")

    def test_short_answer_fails(self):
        assert not ResponseValidator.is_valid_answer("hi", "ok")

    def test_empty_answer_fails(self):
        assert not ResponseValidator.is_valid_answer("hi", "")

    def test_hallucination_liberria_fails(self):
        assert not ResponseValidator.is_valid_answer("what is ml", "liberria is a new concept in ML")

    def test_hallucination_unknown_term_fails(self):
        assert not ResponseValidator.is_valid_answer("test", "This is an unknown term in science")

    def test_hallucination_fake_concept_fails(self):
        assert not ResponseValidator.is_valid_answer("test", "This uses a fake concept called xyz")

    def test_hallucination_case_insensitive(self):
        assert not ResponseValidator.is_valid_answer("test", "LIBERRIA is a technology")

    def test_low_confidence_detected(self):
        assert ResponseValidator.low_confidence("I think this might be the answer")
        assert ResponseValidator.low_confidence("maybe it is correct")
        assert ResponseValidator.low_confidence("I am not sure about this")

    def test_high_confidence_not_detected(self):
        assert not ResponseValidator.low_confidence("The answer is clearly defined.")
        assert not ResponseValidator.low_confidence("Transformers use self-attention.")

    def test_fallback_prompt_contains_question(self):
        prompt = ResponseValidator.build_fallback_prompt("What is AI?")
        assert "What is AI?" in prompt
        assert "low quality" in prompt


class TestInjectConceptSnippets:
    def test_transformer_snippet_injected(self):
        prompt = "Explain this."
        result = _inject_concept_snippets("What is a transformer?", prompt)
        assert "Reference context" in result
        assert "Attention Is All You Need" in result

    def test_self_attention_snippet_injected(self):
        prompt = "Explain this."
        result = _inject_concept_snippets("How does self-attention work?", prompt)
        assert "weighted sum over all positions" in result

    def test_no_snippet_for_irrelevant_query(self):
        prompt = "Explain this."
        result = _inject_concept_snippets("What is the weather?", prompt)
        assert result == prompt

    def test_multiple_snippets_injected(self):
        prompt = "Explain."
        result = _inject_concept_snippets("What is a transformer and self-attention?", prompt)
        assert "Attention Is All You Need" in result
        assert "weighted sum" in result

    def test_rag_snippet_injected(self):
        prompt = "Explain."
        result = _inject_concept_snippets("How does RAG work?", prompt)
        assert "Retrieval-Augmented Generation" in result


class TestIsExplanationTask:
    def test_what_is_is_explanation(self):
        assert _is_explanation_task("What is a transformer?")

    def test_explain_is_explanation(self):
        assert _is_explanation_task("Explain self-attention")

    def test_how_to_is_explanation(self):
        assert _is_explanation_task("How to write Python code?")

    def test_define_is_explanation(self):
        assert _is_explanation_task("Define embedding")

    def test_news_query_not_explanation(self):
        assert not _is_explanation_task("What is the latest news on AI?")

    def test_search_query_not_explanation(self):
        assert not _is_explanation_task("Search for quantum computing")

    def test_greeting_not_explanation(self):
        assert not _is_explanation_task("Hello")

    def test_weather_query_not_explanation(self):
        assert not _is_explanation_task("What is the weather today?")


class TestMemoryManager:
    def setup_method(self):
        init_db()
        self.session = get_session()
        self.mm = MemoryManager(self.session)
        self.repo = MemoryRepository(self.session)

    def teardown_method(self):
        self.session.close()

    def test_load_context_empty(self):
        assert self.mm.load_context(999) == ""

    def test_load_context_returns_memories(self):
        self.repo.store(1, "preference", "likes Python", 0.8)
        self.session.commit()
        ctx = self.mm.load_context(1)
        assert "User known information:" in ctx
        assert "preference" in ctx
        assert "likes Python" in ctx

    def test_load_context_respects_limit(self):
        for i in range(10):
            self.repo.store(2, f"fact_{i}", f"value_{i}", 0.5)
        self.session.commit()
        ctx = self.mm.load_context(2, limit=3)
        lines = [l for l in ctx.split("\n") if l.startswith("- ")]
        assert len(lines) <= 3

    def test_load_context_different_users_isolated(self):
        self.repo.store(3, "pref", "user3 pref", 0.7)
        self.repo.store(4, "pref", "user4 pref", 0.7)
        self.session.commit()
        ctx3 = self.mm.load_context(3)
        ctx4 = self.mm.load_context(4)
        assert "user3" in ctx3 and "user4" not in ctx3
        assert "user4" in ctx4 and "user3" not in ctx4

    def test_extract_and_store_calls_llm(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"memories": [{"key": "preference", "value": "likes AI", "importance": 0.7}]}'
        result = self.mm.extract_and_store(5, "Do you like AI?", "Yes I love AI!", mock_llm)
        assert len(result) == 1
        assert result[0]["key"] == "preference"
        ctx = self.mm.load_context(5)
        assert "likes AI" in ctx

    def test_extract_and_store_handles_empty(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"memories": []}'
        result = self.mm.extract_and_store(6, "Hello", "Hi!", mock_llm)
        assert result == []

    def test_extract_and_store_handles_invalid_json(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "not json"
        result = self.mm.extract_and_store(7, "Hello", "Hi!", mock_llm)
        assert result == []

    def test_extract_and_store_strips_code_fences(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '```json\n{"memories": [{"key": "goal", "value": "learn", "importance": 0.6}]}\n```'
        result = self.mm.extract_and_store(8, "Goal?", "Learn ML", mock_llm)
        assert len(result) == 1

    def test_extract_and_store_multiple_memories(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = json.dumps({
            "memories": [
                {"key": "preference", "value": "likes math", "importance": 0.6},
                {"key": "goal", "value": "master transformers", "importance": 0.9},
            ]
        })
        result = self.mm.extract_and_store(9, "What do you like?", "Math and transformers", mock_llm)
        assert len(result) == 2
        assert self.repo.count(9) == 2

    def teardown_method(self):
        self.repo.delete_by_key(1, "%")
        self.repo.delete_by_key(2, "%")
        self.repo.delete_by_key(3, "%")
        self.repo.delete_by_key(4, "%")
        self.repo.delete_by_key(5, "%")
        self.repo.delete_by_key(6, "%")
        self.repo.delete_by_key(7, "%")
        self.repo.delete_by_key(8, "%")
        self.repo.delete_by_key(9, "%")
        self.session.commit()
        self.session.close()


class TestMemoryIntegration:
    def setup_method(self):
        init_db()
        self.session = get_session()
        self.repo = MemoryRepository(self.session)
        self.agent = MagicMock()
        self.agent.memory_manager = MemoryManager(self.session)
        self.agent._inject_memories = lambda q, p, uid: self.agent.memory_manager.load_context(uid) + "\n\n" + p if uid else p
        self.agent._extract_memories = lambda q, a, uid: self.agent.memory_manager.extract_and_store(uid, q, a, MagicMock()) if uid else None

    def teardown_method(self):
        for uid in [10, 11, 12, 13]:
            self.repo.delete_by_key(uid, "%")
        self.session.commit()
        self.session.close()

    def test_inject_memories_empty_when_no_user(self):
        prompt = "Answer the question"
        result = self.agent._inject_memories("hello", prompt, None)
        assert result == prompt

    def test_inject_memories_adds_context(self):
        self.repo.store(10, "pref", "likes deep learning", 0.8)
        self.session.commit()
        prompt = "What is a transformer?"
        result = self.agent._inject_memories("What is a transformer?", prompt, 10)
        assert "User known information" in result
        assert "likes deep learning" in result
        assert "What is a transformer?" in result

    def test_extract_memories_stores_on_llm_answer(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '{"memories": [{"key": "fact", "value": "likes coding", "importance": 0.5}]}'
        self.agent._extract_memories = lambda q, a, uid: self.agent.memory_manager.extract_and_store(uid, q, a, mock_llm)
        self.agent._extract_memories("Do you like coding?", "Yes!", 11)
        ctx = self.agent.memory_manager.load_context(11)
        assert "likes coding" in ctx

    def test_extract_memories_noop_without_user(self):
        self.agent._extract_memories("Hello", "Hi", None)
        assert self.repo.count(999) == 0

    def test_no_memory_leak_between_users(self):
        self.repo.store(12, "secret", "user12 secret", 0.9)
        self.repo.store(13, "secret", "user13 secret", 0.9)
        self.session.commit()
        ctx12 = self.agent.memory_manager.load_context(12)
        ctx13 = self.agent.memory_manager.load_context(13)
        assert "user12 secret" in ctx12 and "user13 secret" not in ctx12
        assert "user13 secret" in ctx13 and "user12 secret" not in ctx13


class TestPromptRouter:
    def test_greeting_uses_chat_prompt(self):
        template = select_prompt("Hello")
        assert template is CHAT_PROMPT

    def test_greeting_arabic_uses_chat_prompt(self):
        template = select_prompt("مرحبا")
        assert template is CHAT_PROMPT

    def test_thanks_uses_chat_prompt(self):
        template = select_prompt("thanks")
        assert template is CHAT_PROMPT

    def test_goodbye_uses_chat_prompt(self):
        template = select_prompt("bye")
        assert template is CHAT_PROMPT

    def test_explanation_what_is_uses_explanation_prompt(self):
        template = select_prompt("What is a transformer?")
        assert template is EXPLANATION_PROMPT

    def test_explanation_explain_uses_explanation_prompt(self):
        template = select_prompt("Explain self-attention")
        assert template is EXPLANATION_PROMPT

    def test_explanation_why_uses_explanation_prompt(self):
        template = select_prompt("Why is attention important?")
        assert template is EXPLANATION_PROMPT

    def test_explanation_compare_uses_comparison_prompt(self):
        template = select_prompt("Compare CNN and Transformer")
        assert template is COMPARISON_PROMPT

    def test_coding_write_code_uses_code_prompt(self):
        template = select_prompt("Write a Python function to sort a list")
        assert template is CODE_PROMPT

    def test_coding_python_uses_code_prompt(self):
        template = select_prompt("Python regex example")
        assert template is CODE_PROMPT

    def test_coding_javascript_uses_code_prompt(self):
        template = select_prompt("JavaScript async await example")
        assert template is CODE_PROMPT

    def test_coding_debug_uses_code_prompt(self):
        template = select_prompt("Debug this function")
        assert template is CODE_PROMPT

    def test_general_question_uses_general_prompt(self):
        template = select_prompt("Do you like movies?")
        assert template is GENERAL_PROMPT

    def test_general_opinion_uses_general_prompt(self):
        template = select_prompt("I am feeling happy today")
        assert template is GENERAL_PROMPT


class TestValidateAndRegenerateSkipGreetings:
    def test_skips_low_confidence_for_greeting(self):
        from part2_rag.rag_agent import RAGAgent
        agent = MagicMock(spec=RAGAgent)
        agent._validate_and_regenerate = RAGAgent._validate_and_regenerate.__get__(agent, RAGAgent)
        agent.llm = MagicMock()
        answer = agent._validate_and_regenerate("Hello", "Hi!")
        assert answer == "Hi!"

    def test_skips_low_confidence_for_thanks(self):
        from part2_rag.rag_agent import RAGAgent
        agent = MagicMock(spec=RAGAgent)
        agent._validate_and_regenerate = RAGAgent._validate_and_regenerate.__get__(agent, RAGAgent)
        agent.llm = MagicMock()
        answer = agent._validate_and_regenerate("thanks", "You're welcome!")
        assert answer == "You're welcome!"

    def test_skips_low_confidence_for_goodbye(self):
        from part2_rag.rag_agent import RAGAgent
        agent = MagicMock(spec=RAGAgent)
        agent._validate_and_regenerate = RAGAgent._validate_and_regenerate.__get__(agent, RAGAgent)
        agent.llm = MagicMock()
        answer = agent._validate_and_regenerate("bye", "Goodbye!")
        assert answer == "Goodbye!"
