"""
Pure Python BM25 (Okapi BM25) implementation.
No external dependencies beyond standard library.
"""
import math
import re
from collections import Counter
from typing import Optional


_STOP_WORDS: set[str] = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with",
    "by", "from", "as", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "this", "that", "these", "those", "it", "its", "it's",
    "and", "or", "but", "not", "no", "nor", "so", "if", "then",
    "than", "too", "very", "just", "about", "also", "more",
    "some", "any", "each", "every", "all", "both", "few", "most",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "i", "me", "my", "myself", "we", "our", "ours", "us",
    "you", "your", "yours", "he", "him", "his", "she", "her",
    "hers", "it", "its", "they", "them", "their", "theirs",
}


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"[a-zA-Z]\w+(?:[-/]\w+)*", text)
    return [t for t in tokens if len(t) > 1 and t not in _STOP_WORDS]


class BM25Okapi:
    def __init__(self, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self._docs: list[str] = []
        self._doc_tokens: list[list[str]] = []
        self._doc_freqs: list[Counter] = []
        self._idf: dict[str, float] = {}
        self._avg_doc_len: float = 0.0
        self._total_docs: int = 0
        self._built = False

    @property
    def built(self) -> bool:
        return self._built

    def index(self, documents: list[str]):
        self._docs = list(documents)
        self._doc_tokens = [_tokenize(d) for d in self._docs]
        self._doc_freqs = [Counter(tokens) for tokens in self._doc_tokens]
        self._total_docs = len(self._docs)
        self._avg_doc_len = sum(len(t) for t in self._doc_tokens) / max(1, self._total_docs)

        df: dict[str, int] = {}
        for freqs in self._doc_freqs:
            for term in freqs:
                df[term] = df.get(term, 0) + 1

        self._idf = {}
        for term, doc_freq in df.items():
            idf = math.log(1 + (self._total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            self._idf[term] = idf

        self._built = True

    def get_scores(self, query: str) -> list[float]:
        if not self._built:
            return []
        query_tokens = _tokenize(query)
        if not query_tokens:
            return [0.0] * self._total_docs

        scores = [0.0] * self._total_docs
        for term in query_tokens:
            idf = self._idf.get(term, 0.0)
            if idf == 0.0:
                continue
            for i, freqs in enumerate(self._doc_freqs):
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                doc_len = sum(freqs.values())
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self._avg_doc_len)
                scores[i] += idf * numerator / denominator

        return scores

    def score(self, query: str, doc_index: int) -> float:
        scores = self.get_scores(query)
        if 0 <= doc_index < len(scores):
            return scores[doc_index]
        return 0.0

    def rank(self, query: str, k: Optional[int] = None) -> list[tuple[int, float]]:
        scores = self.get_scores(query)
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: -x[1])
        if k is not None:
            indexed = indexed[:k]
        return indexed
