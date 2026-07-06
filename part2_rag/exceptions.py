class RAGError(Exception):
    """Base exception for RAG module."""

class EmbeddingError(RAGError):
    """Embedding operation failed."""

class VectorstoreError(RAGError):
    """Vectorstore operation failed."""

class RetrievalError(RAGError):
    """Document retrieval failed."""

class LLMError(RAGError):
    """LLM invocation failed."""

class APIError(RAGError):
    """API request error."""
