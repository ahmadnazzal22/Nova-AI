import os
import random
import torch
from torch.utils.data import Dataset
from .exceptions import DatasetError
from .tokenizer import WordTokenizer
from .logger import get_logger

logger = get_logger(__name__)


def load_texts(path: str) -> list[str]:
    if not os.path.exists(path):
        raise DatasetError(f"Data file not found: {path}")
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and len(line) >= 3:
                texts.append(line)
    if not texts:
        raise DatasetError(f"No valid lines found in {path}")
    logger.info("Loaded %d lines from %s", len(texts), path)
    return texts


class TextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer: WordTokenizer, max_len: int):
        self.src_data: list[torch.Tensor] = []
        self.tgt_data: list[torch.Tensor] = []
        pad_id = tokenizer.word2idx["<PAD>"]

        for text in texts:
            tokens = tokenizer.encode(text, max_len)
            src = tokens[:-1]
            tgt = tokens[1:]
            if len(src) < max_len:
                src = src + [pad_id] * (max_len - len(src))
                tgt = tgt + [pad_id] * (max_len - len(tgt))
            self.src_data.append(torch.tensor(src[:max_len], dtype=torch.long))
            self.tgt_data.append(torch.tensor(tgt[:max_len], dtype=torch.long))

    def __len__(self) -> int:
        return len(self.src_data)

    def __getitem__(self, idx: int):
        return self.src_data[idx], self.tgt_data[idx]


def create_train_val_datasets(
    data_path: str,
    tokenizer: WordTokenizer,
    max_len: int,
    val_split: float,
    shuffle: bool = True,
    seed: int = 42,
) -> tuple[TextDataset, TextDataset]:
    texts = load_texts(data_path)
    if shuffle:
        random.seed(seed)
        random.shuffle(texts)

    split_idx = int(len(texts) * (1 - val_split))
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]

    logger.info("Train samples: %d | Val samples: %d", len(train_texts), len(val_texts))

    train_ds = TextDataset(train_texts, tokenizer, max_len)
    val_ds = TextDataset(val_texts, tokenizer, max_len)
    return train_ds, val_ds
