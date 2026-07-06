import torch
import torch.nn as nn
from .embeddings import TokenEmbedding
from .positional_encoding import PositionalEncoding
from .attention import MultiHeadAttention
from .feed_forward import FeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None, kv_cache=None, layer_idx=0):
        attn_out = self.self_attn(x, x, x, tgt_mask, kv_cache=kv_cache, cache_prefix=f"self_{layer_idx}")
        x = self.norm1(x + self.dropout(attn_out))
        attn_out = self.cross_attn(x, enc_output, enc_output, src_mask, kv_cache=kv_cache, cache_prefix=f"cross_{layer_idx}")
        x = self.norm2(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout=0.1):
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout=0.1):
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None, kv_cache=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for i, layer in enumerate(self.layers):
            x = layer(x, enc_output, src_mask, tgt_mask, kv_cache=kv_cache, layer_idx=i)
        return self.norm(x)

    def forward_cached(self, x, enc_output, src_mask, kv_cache, pos_offset):
        x = self.embedding(x)
        x = self.pos_encoding(x, offset=pos_offset)
        for i, layer in enumerate(self.layers):
            x = layer(x, enc_output, src_mask, tgt_mask=None, kv_cache=kv_cache, layer_idx=i)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            num_layers=config.num_encoder_layers,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )
        self.decoder = Decoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            num_layers=config.num_decoder_layers,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )
        self.fc_out = nn.Linear(config.d_model, config.vocab_size)
        self._nopeak_cache: dict[int, torch.Tensor] = {}

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        seq_len = tgt.size(1)
        if seq_len not in self._nopeak_cache:
            self._nopeak_cache[seq_len] = torch.tril(
                torch.ones(seq_len, seq_len, device=tgt.device)
            ).bool()
        nopeak_mask = self._nopeak_cache[seq_len]
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        return self.fc_out(dec_output)

    def encode(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return self.encoder(src, src_mask)

    @torch.no_grad()
    def generate(self, src, max_len, sos_idx, eos_idx, temperature=1.0):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        enc_output = self.encoder(src, src_mask)
        kv_cache = {}
        batch_size = src.size(0)

        max_pos = self.decoder.pos_encoding.pe.size(1)
        max_len = min(max_len, max_pos - 1)

        tgt = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=src.device)
        pos = 0

        for _ in range(max_len):
            dec_output = self.decoder.forward_cached(
                tgt[:, -1:], enc_output, src_mask, kv_cache=kv_cache, pos_offset=pos
            )
            logits = self.fc_out(dec_output[:, -1, :])

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            tgt = torch.cat([tgt, next_token], dim=1)
            pos += 1

            if next_token.item() == eos_idx:
                break

        return tgt[:, 1:]
