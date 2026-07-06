import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .config import TransformerConfig
from .tokenizer import WordTokenizer
from .transformer import Transformer
from .dataset import create_train_val_datasets, load_texts
from .logger import get_logger

logger = get_logger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(config: TransformerConfig | None = None, data_path: str | None = None) -> tuple[Transformer, WordTokenizer, TransformerConfig]:
    config = config or TransformerConfig()
    data_path = data_path or config.train_data_path
    set_seed(config.seed)

    logger.info("Starting training | device=%s | d_model=%d | heads=%d | layers=%d | epochs=%d | seed=%d",
                config.device, config.d_model, config.num_heads, config.num_encoder_layers, config.epochs, config.seed)

    tokenizer = WordTokenizer(
        tokenizer_type=config.tokenizer_type,
        max_vocab_size=config.max_vocab_size,
    )

    texts = load_texts(data_path)
    tokenizer.fit(texts)
    config.vocab_size = tokenizer.vocab_size

    train_ds, val_ds = create_train_val_datasets(
        data_path=data_path,
        tokenizer=tokenizer,
        max_len=config.max_seq_len,
        val_split=config.val_split,
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False) if len(val_ds) > 0 else None

    model = Transformer(config).to(config.device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word2idx["<PAD>"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9)

    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        for src, tgt in train_loader:
            src, tgt = src.to(config.device), tgt.to(config.device)
            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])
            loss = criterion(output.contiguous().view(-1, config.vocab_size), tgt[:, 1:].contiguous().view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for src, tgt in val_loader:
                    src, tgt = src.to(config.device), tgt.to(config.device)
                    output = model(src, tgt[:, :-1])
                    loss = criterion(output.contiguous().view(-1, config.vocab_size), tgt[:, 1:].contiguous().view(-1))
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)

            if epoch == 1 or epoch % 5 == 0:
                logger.info("Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f", epoch, config.epochs, avg_train_loss, avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                _save_checkpoint(model, tokenizer, config, is_best=True)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping at epoch %d (val_loss=%.4f)", epoch, avg_val_loss)
                    break
        else:
            if epoch == 1 or epoch % 10 == 0:
                logger.info("Epoch %3d/%d | train_loss=%.4f", epoch, config.epochs, avg_train_loss)

    _save_checkpoint(model, tokenizer, config, is_best=False)
    logger.info("Training complete | best_val_loss=%.4f", best_val_loss)
    return model, tokenizer, config


def _save_checkpoint(model: Transformer, tokenizer: WordTokenizer, config: TransformerConfig, is_best: bool = False):
    path = "transformer_checkpoint.pth" if not is_best else "transformer_best.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "tokenizer": tokenizer,
    }, path)
    logger.info("Checkpoint saved: %s", path)


def extract_embeddings(model: Transformer, tokenizer: WordTokenizer, config: TransformerConfig):
    embeddings = model.encoder.embedding.embedding.weight.detach().cpu().numpy()
    torch.save({
        "embeddings": embeddings,
        "vocab_size": config.vocab_size,
        "d_model": config.d_model,
        "word2idx": tokenizer.word2idx,
        "idx2word": tokenizer.idx2word,
        "tokenizer_type": tokenizer.tokenizer_type,
    }, "embeddings.pth")
    logger.info("Embeddings saved: shape=%s", embeddings.shape)
    return embeddings


if __name__ == "__main__":
    model, tokenizer, config = train()
    extract_embeddings(model, tokenizer, config)
