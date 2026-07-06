import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import pytest
from part2_rag.rag_agent import RAGAgent, MockLLM, _clean_text


class TestMockLLM:
    def test_llm_type(self):
        llm = MockLLM()
        assert llm._llm_type == "mock"

    def test_call_returns_clean_summary(self):
        llm = MockLLM()
        prompt = "CONTEXT:\nTransformers use self-attention.\n\nQUESTION:\nWhat is a transformer?\n\nANSWER:"
        result = llm._call(prompt)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Based on the provided information" in result

    def test_call_no_context(self):
        llm = MockLLM()
        prompt = "CONTEXT:\nNo context available.\n\nQUESTION:\nWhat?\n\nANSWER:"
        result = llm._call(prompt)
        assert "don't have enough" in result

    def test_identifying_params(self):
        llm = MockLLM()
        assert llm._identifying_params == {"model": "mock"}


class TestTextCleaner:
    def test_collapses_repeated_words(self):
        assert _clean_text("word word word word") == "word"

    def test_removes_excessive_whitespace(self):
        assert _clean_text("hello     world") == "hello world"

    def test_removes_repeated_chars(self):
        result = _clean_text("aaaa")
        assert len(result) < 4

    def test_handles_empty_string(self):
        assert _clean_text("   ") == ""


class TestRAGAgent:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        ckpt = "transformer_checkpoint.pth"
        if not os.path.exists(ckpt):
            pytest.skip("No checkpoint found")
        RAGAgent._instance = None
        self.tmpdir = tempfile.mkdtemp()
        self.persist_dir = self.tmpdir
        yield
        import shutil
        for _ in range(3):
            try:
                shutil.rmtree(self.tmpdir)
                break
            except PermissionError:
                import time
                time.sleep(0.5)

    def test_init(self):
        agent = RAGAgent(persist_dir=self.persist_dir, llm=MockLLM())
        assert agent.db is not None

    def test_add_documents_cleans_text(self):
        agent = RAGAgent(persist_dir=self.persist_dir, llm=MockLLM())
        count = agent.add_documents(["word word word word Transformer uses self-attention."])
        assert count >= 1

    def test_query_returns_answer_and_sources(self):
        agent = RAGAgent(persist_dir=self.persist_dir, llm=MockLLM())
        agent.add_documents([
            "Transformers use self-attention to process sequences in parallel.",
            "Attention helps models focus on important parts of the input.",
        ])
        result = agent.query("Search for self-attention in the documents")
        assert "answer" in result
        assert "sources" in result

    def test_query_empty_string(self):
        agent = RAGAgent(persist_dir=self.persist_dir, llm=MockLLM())
        result = agent.query("")
        assert "answer" in result
        assert "sources" in result

    def test_deduplicates_sources(self):
        agent = RAGAgent(persist_dir=self.persist_dir, llm=MockLLM())
        agent.add_documents(["Same document content.", "Same document content."])
        result = agent.query("Search for document content")
        unique = list(dict.fromkeys(result["sources"]))
        assert len(unique) == len(result["sources"])
