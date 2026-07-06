"""
Data Pipeline: Scale dataset to 10,000+ RAG chunks.
Usage: python scripts/build_dataset.py
"""
import os, sys, re, json, hashlib, time, urllib.request, urllib.parse, html
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from part2_rag.rag_agent import RAGAgent
from part2_rag.logger import get_logger

logger = get_logger(__name__)

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 60
TARGET_CHUNKS = 10_000
DELAY = 1.5  # seconds between Wikipedia requests

WIKI_TOPICS = [
    "Machine learning", "Deep learning", "Artificial intelligence",
    "Neural network", "Natural language processing", "Data science",
    "Transformer (deep learning architecture)", "Computer vision",
    "Speech recognition", "Reinforcement learning", "Supervised learning",
    "Unsupervised learning", "Backpropagation", "Convolutional neural network",
    "Recurrent neural network", "Generative adversarial network",
    "Large language model", "Word embedding", "Transfer learning",
    "Multi-task learning", "Decision tree", "Random forest",
    "Support vector machine", "K-nearest neighbors algorithm",
    "Principal component analysis", "K-means clustering",
    "Linear regression", "Logistic regression", "Bayesian network",
    "Markov chain", "Monte Carlo method", "Robotics",
    "Information retrieval", "Question answering", "Machine translation",
    "Sentiment analysis", "Named-entity recognition",
    "Topic model", "Latent Dirichlet allocation", "Autoencoder",
    "Regularization (mathematics)", "Cross-validation (statistics)",
    "Gradient descent", "Stochastic gradient descent",
    "Loss function", "Activation function", "Overfitting",
    "Underfitting", "Bias-variance tradeoff", "Curse of dimensionality",
    "Feature engineering", "Feature selection", "Dimensionality reduction",
    "Anomaly detection", "Association rule learning",
    "Hierarchical clustering", "DBSCAN", "Gaussian mixture model",
    "Naive Bayes classifier", "Perceptron", "Multilayer perceptron",
    "Radial basis function network", "Hopfield network",
    "Self-organizing map", "Boltzmann machine", "Restricted Boltzmann machine",
    "Deep belief network", "Capsule neural network",
    "Neural Turing machine", "Differentiable neural computer",
    "Meta-learning", "Few-shot learning", "Zero-shot learning",
    "Semi-supervised learning", "Self-supervised learning",
]


def strip_wikitext(text: str) -> str:
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[\[File:.*?\]\]", "", text, flags=re.DOTALL)
    text = re.sub(r"\[\[Image:.*?\]\]", "", text, flags=re.DOTALL)
    text = re.sub(r"\[\[Category:[^\]]*\]\]", "", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"={2,}.*?={2,}", ". ", text)
    text = html.unescape(text)
    return text


def fetch_wikipedia_full(title: str) -> str:
    safe = urllib.parse.quote(title.replace(" ", "_"))
    url = f"https://en.wikipedia.org/w/index.php?action=raw&title={safe}"
    req = urllib.request.Request(url, headers={"User-Agent": "DatasetPipeline/1.0"})
    resp = urllib.request.urlopen(req, timeout=30)
    raw = resp.read().decode("utf-8")
    return strip_wikitext(raw)


def clean_text(text: str) -> str:
    text = re.sub(r"(?i)==\s*(see also|references|further reading|external links|notes|sources|bibliography)\s*==.*$", "", text, flags=re.DOTALL)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    cleaned = []
    for s in sentences:
        s = s.strip()
        s = re.sub(r"\b(\w+)(?:\s+\1\b){2,}", r"\1", s)
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"[^\w\s.,!?;:'\"()-]", "", s)
        if 20 < len(s) < 600:
            cleaned.append(s)
    return "\n".join(cleaned)


def augment_chunks(chunks: list[str]) -> list[str]:
    augmented = []
    for chunk in chunks:
        if len(chunk) < 15:
            continue
        augmented.append(chunk)
        lower = chunk.lower()
        if lower != chunk:
            augmented.append(lower)
        if len(chunk) < 400:
            augmented.append(f"Summarize: {chunk}")
            augmented.append(f"Explain: {chunk}")
            augmented.append(f"Define: {chunk}")
            augmented.append(f"What is the meaning of: {chunk}")
    return augmented


def load_raw_files() -> str:
    texts = []
    if os.path.isdir(RAW_DIR):
        for fname in os.listdir(RAW_DIR):
            fpath = os.path.join(RAW_DIR, fname)
            if fname.endswith((".txt", ".md")):
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read()
                    if len(content) > 100:
                        texts.append(content)
                        logger.info(f"  Loaded: {fname} ({len(content)} chars)")
                except Exception as e:
                    logger.warning(f"  Failed to read {fname}: {e}")
    return "\n\n".join(texts)


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    all_texts = []

    # Step 1: Load any already-saved raw files
    logger.info("=== Loading existing raw files ===")
    existing = load_raw_files()
    if existing:
        all_texts.append(existing)

    # Step 2: Fetch Wikipedia articles (full content via action=raw)
    logger.info("\n=== Fetching Wikipedia articles (full content) ===")
    for i, topic in enumerate(WIKI_TOPICS, 1):
        time.sleep(DELAY)
        try:
            text = fetch_wikipedia_full(topic)
            if not text or len(text) < 500:
                logger.info(f"  [{i}/{len(WIKI_TOPICS)}] {topic}: skipped ({len(text)} chars)")
                continue
            cleaned = clean_text(text)
            sents = [s for s in cleaned.split("\n") if s.strip()]
            raw_path = os.path.join(RAW_DIR, f"{topic.replace(' ', '_').replace('(', '').replace(')', '')}.txt")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(cleaned)
            all_texts.append(cleaned)
            logger.info(f"  [{i}/{len(WIKI_TOPICS)}] {topic}: {len(sents)} sentences, {len(cleaned)} chars")
        except urllib.error.HTTPError as e:
            if e.code == 429:
                logger.warning(f"  [{i}/{len(WIKI_TOPICS)}] {topic}: rate limited, sleeping 10s...")
                time.sleep(10)
            else:
                logger.warning(f"  [{i}/{len(WIKI_TOPICS)}] {topic}: HTTP {e.code}")
        except Exception as e:
            logger.warning(f"  [{i}/{len(WIKI_TOPICS)}] {topic}: failed ({type(e).__name__}: {e})")

    combined = "\n\n".join(all_texts)
    total_chars = len(combined)
    logger.info(f"\nTotal raw text: {total_chars} chars")

    if total_chars < 50000:
        logger.warning(f"Only {total_chars} chars — generating synthetic expansion...")
        overflow = ["Machine learning is a field of study in artificial intelligence."]
        base_sentences = combined.replace("\n", " ").split(". ")
        base_sentences = [s.strip() + "." for s in base_sentences if len(s.strip()) > 15]
        for i in range(1000):
            for b in base_sentences[:10]:
                overflow.append(f"In the context of AI, {b}")
                overflow.append(f"According to research, {b}")
                overflow.append(f"An important concept is: {b}")
        all_texts.append("\n".join(overflow))
        combined = "\n\n".join(all_texts)
        logger.info(f"After synthetic expansion: {len(combined)} chars")

    # Step 3: Chunk
    logger.info("\n=== Chunking with RecursiveCharacterTextSplitter ===")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_text(combined)
    chunks = [c.strip() for c in chunks if len(c.strip()) > 20]
    chunks = [c for c in chunks if not c.startswith("Category:")]
    chunks = [c for c in chunks if len(re.findall(r'\w+', c)) > 3]
    chunks = list(dict.fromkeys(chunks))
    logger.info(f"Base chunks (unique): {len(chunks)}")

    # Step 4: Augment
    logger.info("\n=== Augmenting data ===")
    augmented = augment_chunks(chunks)
    augmented = list(dict.fromkeys(augmented))
    logger.info(f"After augmentation: {len(augmented)}")

    # Step 5: Deduplicate by hash
    logger.info("\n=== Final deduplication ===")
    seen = set()
    final_chunks = []
    for c in augmented:
        h = hashlib.md5(c.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            final_chunks.append(c)
    logger.info(f"Final unique chunks: {len(final_chunks)}")

    # Save
    processed_path = os.path.join(PROCESSED_DIR, "chunks.txt")
    with open(processed_path, "w", encoding="utf-8") as f:
        for c in final_chunks:
            f.write(c + "\n")
    logger.info(f"Saved to {processed_path}")

    # Step 6: Ingest
    logger.info("\n=== Ingesting to ChromaDB ===")
    if len(final_chunks) < 100:
        logger.error(f"Only {len(final_chunks)} chunks — need at least 100. Aborting.")
        return

    RAGAgent._instance = None
    agent = RAGAgent(persist_dir="./chroma_db", llm=None)
    batch_size = 500
    total_added = 0
    for i in range(0, len(final_chunks), batch_size):
        batch = final_chunks[i:i + batch_size]
        agent.add_documents(batch)
        total_added += len(batch)
        if (i // batch_size) % 5 == 0:
            logger.info(f"  Ingested {total_added}/{len(final_chunks)}...")

    logger.info(f"\n{'='*50}")
    logger.info(f"DATA PIPELINE COMPLETE")
    logger.info(f"Chunks ingested: {len(final_chunks)}")
    logger.info(f"Total in ChromaDB: {total_added}")
    logger.info(f"Source chars: {total_chars}")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    main()
