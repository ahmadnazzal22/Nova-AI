import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import pytest
from part2_rag.custom_embeddings import TransformerEmbeddings


class TestTransformerEmbeddings:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Find available checkpoint
        for path in ["transformer_best.pth", "transformer_checkpoint.pth"]:
            if os.path.exists(path):
                self.ckpt = path
                break
        else:
            pytest.skip("No checkpoint found — run training first")

    def test_init(self):
        emb = TransformerEmbeddings(checkpoint_path=self.ckpt)
        assert emb.config.d_model > 0
        assert emb.tokenizer.vocab_size > 0

    def test_embed_query_returns_list(self):
        emb = TransformerEmbeddings(checkpoint_path=self.ckpt)
        vec = emb.embed_query("hello world")
        assert isinstance(vec, list)
        assert len(vec) > 0

    def test_embed_query_dimension(self):
        emb = TransformerEmbeddings(checkpoint_path=self.ckpt)
        vec = emb.embed_query("test")
        assert len(vec) == emb.config.d_model

    def test_embed_documents(self):
        emb = TransformerEmbeddings(checkpoint_path=self.ckpt)
        vecs = emb.embed_documents(["doc one", "doc two", "doc three"])
        assert len(vecs) == 3
        assert all(len(v) == emb.config.d_model for v in vecs)

    def test_embed_query_normalized(self):
        emb = TransformerEmbeddings(checkpoint_path=self.ckpt)
        vec = emb.embed_query("test vector")
        norm = math.sqrt(sum(v * v for v in vec))
        assert abs(norm - 1.0) < 1e-5

    def test_embedding_cache(self):
        emb = TransformerEmbeddings(checkpoint_path=self.ckpt)
        v1 = emb.embed_query("cached test")
        v2 = emb.embed_query("cached test")
        assert v1 == v2
