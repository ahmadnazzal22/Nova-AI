import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import torch
from part1_transformer.attention import MultiHeadAttention


class TestMultiHeadAttention:
    def test_output_shape(self):
        attn = MultiHeadAttention(d_model=128, num_heads=4)
        x = torch.randn(2, 10, 128)
        out = attn(x, x, x)
        assert out.shape == (2, 10, 128)

    def test_with_mask(self):
        attn = MultiHeadAttention(d_model=64, num_heads=4)
        x = torch.randn(1, 8, 64)
        mask = torch.ones(1, 1, 1, 8).bool()
        out = attn(x, x, x, mask)
        assert out.shape == (1, 8, 64)

    def test_multi_head_split(self):
        d_model, num_heads = 256, 8
        attn = MultiHeadAttention(d_model, num_heads)
        x = torch.randn(3, 12, d_model)
        out = attn(x, x, x)
        d_k = d_model // num_heads
        assert d_k == 32
        assert out.shape == (3, 12, d_model)

    def test_batch_independence(self):
        attn = MultiHeadAttention(d_model=32, num_heads=2)
        x = torch.randn(4, 6, 32)
        out = attn(x, x, x)
        assert out.shape == (4, 6, 32)

    def test_different_qkv(self):
        attn = MultiHeadAttention(d_model=64, num_heads=4)
        q = torch.randn(2, 8, 64)
        k = torch.randn(2, 12, 64)
        v = torch.randn(2, 12, 64)
        out = attn(q, k, v)
        assert out.shape == (2, 8, 64)
