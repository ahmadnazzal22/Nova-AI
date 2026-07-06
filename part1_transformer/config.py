import os
import torch
from dotenv import load_dotenv

load_dotenv()

class TransformerConfig:
    def __init__(self):
        self.vocab_size = max(10, int(os.getenv("VOCAB_SIZE", "5000")))
        self.d_model = max(16, int(os.getenv("D_MODEL", "256")))
        self.num_heads = max(1, int(os.getenv("NUM_HEADS", "8")))
        self.num_encoder_layers = max(1, int(os.getenv("NUM_ENCODER_LAYERS", "4")))
        self.num_decoder_layers = max(1, int(os.getenv("NUM_DECODER_LAYERS", "4")))
        self.d_ff = max(16, int(os.getenv("D_FF", "512")))
        self.max_seq_len = max(8, int(os.getenv("MAX_SEQ_LEN", "128")))
        self.dropout = min(0.9, max(0.0, float(os.getenv("DROPOUT", "0.1"))))
        self.batch_size = max(1, int(os.getenv("BATCH_SIZE", "32")))
        self.epochs = max(1, int(os.getenv("EPOCHS", "100")))
        self.lr = max(1e-6, float(os.getenv("LEARNING_RATE", "0.001")))
        self.val_split = min(0.5, max(0.0, float(os.getenv("VAL_SPLIT", "0.1"))))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = int(os.getenv("SEED", "42"))

        # --- Data ---
        self.train_data_path = os.getenv("TRAIN_DATA_PATH", "data/sample.txt")

        # --- Tokenizer ---
        self.tokenizer_type = os.getenv("TOKENIZER_TYPE", "word")
        self.max_vocab_size = int(os.getenv("MAX_VOCAB_SIZE", "5000"))

        # --- RAG / Vectorstore ---
        self.chroma_db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        self.chroma_collection = os.getenv("CHROMA_COLLECTION", "langchain")
        self.retrieval_k = int(os.getenv("RETRIEVAL_K", "3"))

        # --- Groq (replaces Ollama when set) ---
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")

        # --- Ollama ---
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
        self.ollama_timeout = int(os.getenv("OLLAMA_TIMEOUT", "60"))
        self.max_model_size_gb = float(os.getenv("MAX_MODEL_SIZE_GB", "5"))

        # --- Chunking ---
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "512"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "64"))

        # --- API ---
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", "8000"))
        self.project_name = os.getenv("PROJECT_NAME", "Custom Transformer RAG API")

    @classmethod
    def from_dict(cls, d: dict) -> "TransformerConfig":
        c = cls.__new__(cls)
        for k, v in d.items():
            setattr(c, k, v)
        return c

    def to_dict(self) -> dict:
        return {k: str(v) if not isinstance(v, (int, float, bool)) else v
                for k, v in self.__dict__.items() if not k.startswith("_")}
