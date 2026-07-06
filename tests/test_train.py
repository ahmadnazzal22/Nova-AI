import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import pytest
from part1_transformer.dataset import load_texts, TextDataset, create_train_val_datasets
from part1_transformer.tokenizer import WordTokenizer


class TestDataset:
    def test_load_texts(self):
        texts = load_texts("data/sample.txt")
        assert len(texts) > 0
        assert all(isinstance(t, str) for t in texts)

    def test_load_texts_missing_file(self):
        with pytest.raises(Exception):
            load_texts("nonexistent.txt")

    def test_text_dataset_length(self):
        tokenizer = WordTokenizer(max_vocab_size=1000)
        tokenizer.fit(["hello world", "test sentence"])
        ds = TextDataset(["hello world", "test sentence"], tokenizer, max_len=10)
        assert len(ds) == 2

    def test_text_dataset_item_shape(self):
        tokenizer = WordTokenizer(max_vocab_size=1000)
        tokenizer.fit(["hello world hello"])
        ds = TextDataset(["hello world hello"], tokenizer, max_len=10)
        src, tgt = ds[0]
        assert src.shape == (10,)
        assert tgt.shape == (10,)

    def test_train_val_split(self):
        tokenizer = WordTokenizer(max_vocab_size=1000)
        texts = load_texts("data/sample.txt")
        tokenizer.fit(texts)
        train_ds, val_ds = create_train_val_datasets(
            "data/sample.txt", tokenizer, max_len=20, val_split=0.2, shuffle=False
        )
        total = len(train_ds) + len(val_ds)
        assert total == len(texts)
        assert len(val_ds) > 0

    def test_train_val_split_zero(self):
        tokenizer = WordTokenizer(max_vocab_size=1000)
        tokenizer.fit(["hello"])
        train_ds, val_ds = create_train_val_datasets(
            "data/sample.txt", tokenizer, max_len=20, val_split=0.0, shuffle=False
        )
        assert len(val_ds) == 0
