import re
from collections import Counter
from .exceptions import TokenizerError
from .logger import get_logger

logger = get_logger(__name__)


class WordTokenizer:
    def __init__(self, tokenizer_type: str = "word", max_vocab_size: int = 5000):
        self.tokenizer_type = tokenizer_type
        self.max_vocab_size = max_vocab_size
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"
        self.special_tokens = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        self._fitted = False

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower().strip()
        tokens = re.findall(r"\b\w+\b|[^\w\s]", text)
        return [t for t in tokens if t.strip()]

    def _build_bpe_vocab(self, texts: list[str]) -> dict[str, int]:
        word_freqs: dict[str, int] = Counter()
        for text in texts:
            for token in self._tokenize(text):
                word_freqs[token] += 1

        base_vocab = set()
        for word in word_freqs:
            base_vocab.update(list(word))
        base_vocab = sorted(base_vocab)
        vocab = {c: i for i, c in enumerate(base_vocab)}

        splits = {word: list(word) for word in word_freqs}
        num_merges = self.max_vocab_size - len(vocab) - len(self.special_tokens)
        num_merges = max(0, min(num_merges, 500))

        for _ in range(num_merges):
            pairs = Counter()
            for word, freq in word_freqs.items():
                symbols = splits[word]
                for i in range(len(symbols) - 1):
                    pairs[(symbols[i], symbols[i + 1])] += freq
            if not pairs:
                break
            best = pairs.most_common(1)[0][0]
            new_token = best[0] + best[1]
            vocab[new_token] = len(vocab)
            for word in word_freqs:
                symbols = splits[word]
                new_symbols = []
                i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and symbols[i] == best[0] and symbols[i + 1] == best[1]:
                        new_symbols.append(new_token)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                splits[word] = new_symbols

        return vocab

    def fit(self, texts: list[str]):
        if self.tokenizer_type == "bpe":
            raw_vocab = self._build_bpe_vocab(texts)
            self.word2idx = {tok: i for i, tok in enumerate(self.special_tokens)}
            for tok, idx in raw_vocab.items():
                self.word2idx[tok] = idx + len(self.special_tokens)
        else:
            word_counts: Counter = Counter()
            for text in texts:
                word_counts.update(self._tokenize(text))
            sorted_words = sorted(word_counts.keys(), key=lambda w: -word_counts[w])
            sorted_words = sorted_words[:self.max_vocab_size]
            self.word2idx = {tok: i for i, tok in enumerate(self.special_tokens + sorted_words)}

        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        self._fitted = True
        logger.info("Tokenizer fitted | type=%s | vocab_size=%d", self.tokenizer_type, self.vocab_size)

    def encode(self, text: str, max_len: int | None = None) -> list[int]:
        if not self._fitted:
            raise TokenizerError("Tokenizer must be fitted before encoding.")
        pad_id = self.word2idx.get(self.pad_token, 0)
        sos_id = self.word2idx.get(self.sos_token, pad_id)
        eos_id = self.word2idx.get(self.eos_token, pad_id)
        unk_id = self.word2idx.get(self.unk_token, pad_id)
        tokens = self._tokenize(text)
        ids = [sos_id]
        for t in tokens:
            ids.append(self.word2idx.get(t, unk_id))
        ids.append(eos_id)
        if max_len is not None and max_len > 0:
            if len(ids) > max_len:
                ids = ids[:max_len]
            elif len(ids) < max_len:
                ids += [pad_id] * (max_len - len(ids))
        return ids

    def decode(self, ids: list[int]) -> str:
        if not self._fitted:
            raise TokenizerError("Tokenizer must be fitted before decoding.")
        tokens = []
        for i in ids:
            w = self.idx2word.get(i, self.unk_token)
            if w in (self.pad_token, self.sos_token, self.eos_token):
                continue
            tokens.append(w)
        return " ".join(tokens)

    def encode_batch(self, texts: list[str], max_len: int) -> list[list[int]]:
        return [self.encode(t, max_len) for t in texts]

    def decode_batch(self, batch_ids: list[list[int]]) -> list[str]:
        return [self.decode(ids) for ids in batch_ids]
