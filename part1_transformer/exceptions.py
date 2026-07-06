class TransformerError(Exception):
    """Base exception for Transformer module."""

class ConfigError(TransformerError):
    """Invalid configuration."""

class TokenizerError(TransformerError):
    """Tokenizer operation failed."""

class DatasetError(TransformerError):
    """Dataset loading or processing failed."""

class TrainingError(TransformerError):
    """Training process failed."""

class ModelLoadError(TransformerError):
    """Model checkpoint loading failed."""

class EmbeddingError(TransformerError):
    """Embedding extraction or application failed."""
