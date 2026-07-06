import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from part1_transformer.embeddings import TokenEmbedding


class TestTokenEmbedding:
    def test_shape(self):
        emb = TokenEmbedding(vocab_size=100, d_model=128)
        x = torch.randint(0, 100, (4, 16))
        out = emb(x)
        assert out.shape == (4, 16, 128)

    def test_scale_factor(self):
        emb = TokenEmbedding(vocab_size=100, d_model=64)
        x = torch.randint(0, 100, (2, 8))
        out = emb(x)
        expected_scale = 64 ** 0.5
        assert abs(out.std() * expected_scale) > 0  # scaled

    def test_different_dims(self):
        emb = TokenEmbedding(vocab_size=50, d_model=256)
        x = torch.randint(0, 50, (1, 10))
        out = emb(x)
        assert out.shape[-1] == 256
