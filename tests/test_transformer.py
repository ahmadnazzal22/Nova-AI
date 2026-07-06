import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from part1_transformer.config import TransformerConfig
from part1_transformer.transformer import Transformer


@pytest.fixture
def tiny_config():
    c = TransformerConfig()
    c.vocab_size = 50
    c.d_model = 32
    c.num_heads = 2
    c.num_encoder_layers = 2
    c.num_decoder_layers = 2
    c.d_ff = 64
    c.max_seq_len = 20
    c.dropout = 0.0
    return c


class TestTransformer:
    def test_forward_shape(self, tiny_config):
        model = Transformer(tiny_config)
        src = torch.randint(0, tiny_config.vocab_size, (4, 16))
        tgt = torch.randint(0, tiny_config.vocab_size, (4, 15))
        out = model(src, tgt)
        assert out.shape == (4, 15, tiny_config.vocab_size)

    def test_encode_shape(self, tiny_config):
        model = Transformer(tiny_config)
        src = torch.randint(0, tiny_config.vocab_size, (2, 12))
        enc = model.encode(src)
        assert enc.shape == (2, 12, tiny_config.d_model)

    def test_generate_mask_shape(self, tiny_config):
        model = Transformer(tiny_config)
        src = torch.randint(0, tiny_config.vocab_size, (2, 10))
        tgt = torch.randint(0, tiny_config.vocab_size, (2, 8))
        src_mask, tgt_mask = model.generate_mask(src, tgt)
        assert src_mask.shape == (2, 1, 1, 10)
        assert tgt_mask.shape == (2, 1, 8, 8)

    def test_mask_is_triangular(self, tiny_config):
        model = Transformer(tiny_config)
        tgt = torch.randint(1, tiny_config.vocab_size, (1, 6))
        _, tgt_mask = model.generate_mask(tgt, tgt)
        assert torch.all(torch.tril(torch.ones(6, 6)) == tgt_mask[0, 0].float())

    def test_gradients_flow(self, tiny_config):
        model = Transformer(tiny_config)
        src = torch.randint(0, tiny_config.vocab_size, (2, 8))
        tgt = torch.randint(0, tiny_config.vocab_size, (2, 7))
        out = model(src, tgt)
        loss = out.sum()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad

    def test_generate_output_shape(self, tiny_config):
        model = Transformer(tiny_config)
        src = torch.randint(1, tiny_config.vocab_size, (1, 8))
        out = model.generate(src, max_len=10, sos_idx=1, eos_idx=2, temperature=0.0)
        assert out.dim() == 2
        assert out.size(0) == 1
        assert out.size(1) <= 10

    def test_generate_with_temperature(self, tiny_config):
        model = Transformer(tiny_config)
        src = torch.randint(1, tiny_config.vocab_size, (1, 8))
        out = model.generate(src, max_len=5, sos_idx=1, eos_idx=2, temperature=0.5)
        assert out.size(1) <= 5

    def test_generate_eos_stops(self, tiny_config):
        model = Transformer(tiny_config)
        src = torch.randint(1, tiny_config.vocab_size, (1, 4))
        out = model.generate(src, max_len=50, sos_idx=1, eos_idx=2, temperature=0.0)
        assert out.size(1) <= 50

    def test_kv_cache_reduces_sequence_length(self, tiny_config):
        model = Transformer(tiny_config)
        src = torch.randint(1, tiny_config.vocab_size, (1, 6))
        enc_out = model.encoder(src, (src != 0).unsqueeze(1).unsqueeze(2))

        pos = 0
        tgt = torch.full((1, 1), 1, dtype=torch.long)

        cache = {}
        out1 = model.decoder.forward_cached(tgt, enc_out, None, kv_cache=cache, pos_offset=pos)
        self_k_keys = [k for k in cache if k.startswith("self_")]
        cross_k_keys = [k for k in cache if k.startswith("cross_")]
        assert len(self_k_keys) == tiny_config.num_decoder_layers * 2, "First step should populate self KV cache"
        assert len(cross_k_keys) == tiny_config.num_decoder_layers * 2, "First step should populate cross KV cache"

        # Second step should retain the same cross cache and extend self cache
        tgt = torch.full((1, 1), 3, dtype=torch.long)
        pos = 1
        out2 = model.decoder.forward_cached(tgt, enc_out, None, kv_cache=cache, pos_offset=pos)
        assert len([k for k in cache if k.startswith("self_")]) == tiny_config.num_decoder_layers * 2
        assert len([k for k in cache if k.startswith("cross_")]) == tiny_config.num_decoder_layers * 2
