import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import pytest
from part1_transformer.tokenizer import WordTokenizer


class TestWordTokenizer:
    def test_fit_and_vocab_size(self):
        tok = WordTokenizer(tokenizer_type="word", max_vocab_size=1000)
        texts = ["hello world", "hello transformer", "test data"]
        tok.fit(texts)
        assert tok.vocab_size > 0
        assert tok.vocab_size <= 1004  # special 4 + words
        assert tok._fitted

    def test_encode_decode_roundtrip(self):
        tok = WordTokenizer(max_vocab_size=1000)
        tok.fit(["hello world", "test sentence here"])
        original = "hello world"
        ids = tok.encode(original, max_len=20)
        decoded = tok.decode(ids)
        assert original in decoded

    def test_encode_with_max_len(self):
        tok = WordTokenizer(max_vocab_size=1000)
        tok.fit(["short text"])
        ids = tok.encode("hello world", max_len=5)
        assert len(ids) == 5

    def test_special_tokens_present(self):
        tok = WordTokenizer(max_vocab_size=1000)
        tok.fit(["hello"])
        for tok_name in [tok.pad_token, tok.sos_token, tok.eos_token, tok.unk_token]:
            assert tok_name in tok.word2idx

    def test_unknown_token(self):
        tok = WordTokenizer(max_vocab_size=10)
        tok.fit(["hello"])
        ids = tok.encode("zzzznotinvocab", max_len=10)
        unk_id = tok.word2idx[tok.unk_token]
        assert unk_id in ids

    def test_encode_batch(self):
        tok = WordTokenizer(max_vocab_size=1000)
        tok.fit(["hello", "world"])
        batch = tok.encode_batch(["hello", "world"], max_len=10)
        assert len(batch) == 2
        assert all(len(ids) == 10 for ids in batch)

    def test_decode_batch(self):
        tok = WordTokenizer(max_vocab_size=1000)
        tok.fit(["hello world"])
        ids = tok.encode_batch(["hello world", "test"], max_len=10)
        decoded = tok.decode_batch(ids)
        assert len(decoded) == 2

    def test_bpe_tokenizer(self):
        tok = WordTokenizer(tokenizer_type="bpe", max_vocab_size=50)
        tok.fit(["hello hello hello world world", "hello world test"])
        assert tok.vocab_size > 4
        ids = tok.encode("hello world", max_len=10)
        decoded = tok.decode(ids)
        assert len(decoded) > 0

    def test_encode_without_fit_raises(self):
        tok = WordTokenizer()
        with pytest.raises(Exception):
            tok.encode("hello")

    def test_empty_text(self):
        tok = WordTokenizer(max_vocab_size=1000)
        tok.fit(["something"])
        ids = tok.encode("", max_len=10)
        assert len(ids) == 10
