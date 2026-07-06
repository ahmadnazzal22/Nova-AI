import os
import numpy as np
import torch
from langchain.embeddings.base import Embeddings
from part1_transformer.transformer import Transformer
from part1_transformer.tokenizer import WordTokenizer
from part1_transformer.config import TransformerConfig
from .exceptions import EmbeddingError
from .logger import get_logger

logger = get_logger(__name__)

_ALLOWED_SAFE_GLOBALS = [TransformerConfig, WordTokenizer]


class TransformerEmbeddings(Embeddings):
    _instances: dict = {}

    def __new__(cls, checkpoint_path=None, pooling="mean"):
        if pooling != "mean":
            logger.warning("Only mean pooling supported; ignoring pooling=%s", pooling)
        key = checkpoint_path or os.getenv("TRANSFORMER_CHECKPOINT", "transformer_best.pth")
        if key not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[key] = instance
        return cls._instances[key]

    def __init__(self, checkpoint_path: str | None = None, pooling: str = "mean"):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self._cache: dict[str, list[float]] = {}

        checkpoint_path = checkpoint_path or os.getenv("TRANSFORMER_CHECKPOINT", "transformer_best.pth")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = "transformer_checkpoint.pth"
        if not os.path.exists(checkpoint_path):
            raise EmbeddingError(f"Checkpoint not found: {checkpoint_path}")

        try:
            with torch.serialization.safe_globals(_ALLOWED_SAFE_GLOBALS):
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception as e:
            raise EmbeddingError(f"Failed to load checkpoint: {e}") from e

        config = checkpoint["config"]
        tokenizer = checkpoint["tokenizer"]
        model = Transformer(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = config.device
        logger.info("Loaded checkpoint: %s | vocab=%d | d_model=%d",
                     checkpoint_path, config.vocab_size, config.d_model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            batch_size = 64
            results = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                token_ids_list = [self.tokenizer.encode(t, self.config.max_seq_len) for t in batch]
                tokens = torch.tensor(token_ids_list, dtype=torch.long, device=self.device)
                with torch.no_grad():
                    emb = self.model.encode(tokens)
                results.extend([self._pool_and_normalize(e) for e in emb])
            return results
        except Exception as e:
            raise EmbeddingError(f"Batch embedding failed for {len(texts)} texts") from e

    def embed_query(self, text: str) -> list[float]:
        if text in self._cache:
            return self._cache[text]
        try:
            token_ids = self.tokenizer.encode(text, self.config.max_seq_len)
            tokens = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            with torch.no_grad():
                emb = self.model.encode(tokens)
            vec = self._pool_and_normalize(emb[0])
            self._cache[text] = vec
            return vec
        except Exception as e:
            raise EmbeddingError(f"Embedding failed for text: {text[:50]}") from e

    @staticmethod
    def _pool_and_normalize(emb: torch.Tensor) -> list[float]:
        avg_emb = emb.mean(dim=0).cpu().numpy()
        norm = np.linalg.norm(avg_emb)
        if norm > 1e-10:
            avg_emb = avg_emb / norm
        return avg_emb.tolist()
