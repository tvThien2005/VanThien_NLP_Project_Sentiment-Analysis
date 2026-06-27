"""
model.py — Transformer Encoder for Sentiment Classification (built from scratch)

Architecture:
  Token Embedding  +  Positional Encoding
         ↓
  N × Transformer Encoder Block
    ├── Multi-Head Self-Attention  (with padding mask)
    ├── Add & LayerNorm
    ├── Feed-Forward Network (FFN)
    └── Add & LayerNorm
         ↓
  Pooling  (mean-pooling over non-PAD tokens)
         ↓
  Classification Head  (Linear → Dropout → Linear)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    EMBED_DIM, NUM_HEADS, NUM_LAYERS, FFN_DIM,
    DROPOUT, MAX_POSITION, NUM_CLASSES, MAX_VOCAB_SIZE
)


# ── 1. Positional Encoding ────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al. 2017).
    Không học được (fixed), giúp model biết vị trí của từng token.
    """
    def __init__(self, embed_dim: int, max_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # pe: [max_len, embed_dim]
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)   # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embed_dim)
        )                                                                        # [embed_dim/2]

        pe[:, 0::2] = torch.sin(position * div_term)   # even indices
        pe[:, 1::2] = torch.cos(position * div_term)   # odd indices

        pe = pe.unsqueeze(0)          # [1, max_len, embed_dim]  — batch dim
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D]"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ── 2. Multi-Head Self-Attention ──────────────────────────────────────────────
class MultiHeadSelfAttention(nn.Module):
    """
    Scaled Dot-Product Multi-Head Attention.
    Tự implement để hiểu rõ từng bước, không dùng nn.MultiheadAttention.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim phải chia hết cho num_heads"

        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.scale      = math.sqrt(self.head_dim)

        # Projection layers
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,               # [B, T, D]
        key_padding_mask: torch.Tensor  # [B, T]  — True ở PAD positions
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape

        # ── Linear projections ───────────────────────────────────────────────
        Q = self.W_q(x)   # [B, T, D]
        K = self.W_k(x)
        V = self.W_v(x)

        # ── Split heads: [B, T, D] → [B, H, T, head_dim] ────────────────────
        def split_heads(t):
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # ── Scaled dot-product attention ──────────────────────────────────────
        # scores: [B, H, T, T]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Mask PAD positions: đặt -inf để softmax → 0
        if key_padding_mask is not None:
            # [B, T] → [B, 1, 1, T]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float("-inf"))

        # Nếu toàn bộ một hàng là -inf (all-PAD row), softmax → NaN → thay bằng 0
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum: [B, H, T, head_dim]
        context = torch.matmul(attn_weights, V)

        # ── Merge heads: [B, H, T, head_dim] → [B, T, D] ────────────────────
        context = context.transpose(1, 2).contiguous().view(B, T, D)
        out = self.W_o(context)                        # [B, T, D]

        return out, attn_weights


# ── 3. Feed-Forward Network ───────────────────────────────────────────────────
class FeedForward(nn.Module):
    """
    Position-wise FFN: Linear → GELU → Dropout → Linear
    GELU tốt hơn ReLU cho NLP tasks.
    """
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── 4. Transformer Encoder Block ──────────────────────────────────────────────
class TransformerEncoderBlock(nn.Module):
    """
    Pre-LayerNorm variant (ổn định hơn Post-LN khi train từ đầu):
      x → LN → MHSA → residual
      x → LN → FFN  → residual
    """
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ffn  = FeedForward(embed_dim, ffn_dim, dropout)
        self.ln1  = nn.LayerNorm(embed_dim)
        self.ln2  = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,                    # [B, T, D]
        key_padding_mask: torch.Tensor,     # [B, T]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self-attention sub-layer (Pre-LN)
        normed = self.ln1(x)
        attn_out, attn_weights = self.attn(normed, key_padding_mask)
        x = x + attn_out                    # residual

        # FFN sub-layer (Pre-LN)
        x = x + self.ffn(self.ln2(x))      # residual

        return x, attn_weights


# ── 5. Full Transformer Sentiment Classifier ──────────────────────────────────
class SentimentTransformer(nn.Module):
    """
    Complete model: Embedding + PE → N×EncoderBlock → Pooling → Classifier
    """
    def __init__(
        self,
        vocab_size: int,
        num_classes: int  = NUM_CLASSES,
        embed_dim: int    = EMBED_DIM,
        num_heads: int    = NUM_HEADS,
        num_layers: int   = NUM_LAYERS,
        ffn_dim: int      = FFN_DIM,
        dropout: float    = DROPOUT,
        max_position: int = MAX_POSITION,
        pad_idx: int      = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # Positional encoding
        self.pos_enc = PositionalEncoding(embed_dim, max_position, dropout)

        # Transformer encoder stack
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # Final LayerNorm (Pre-LN style needs this after last block)
        self.final_ln = nn.LayerNorm(embed_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

        # Weight initialization
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform init cho linear layers, normal cho embeddings."""
        for name, p in self.named_parameters():
            if "embed" in name and p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(
        self,
        input_ids: torch.Tensor,         # [B, T]
        attention_mask: torch.Tensor,    # [B, T]  — True = real token
    ) -> dict:
        # key_padding_mask: True ở PAD (ngược với attention_mask)
        pad_mask = ~attention_mask        # [B, T]

        # Token embedding + positional encoding
        x = self.token_embed(input_ids)   # [B, T, D]
        x = self.pos_enc(x)               # [B, T, D]

        # Pass through encoder stack
        all_attn_weights = []
        for block in self.encoder_blocks:
            x, attn_w = block(x, pad_mask)
            all_attn_weights.append(attn_w)

        x = self.final_ln(x)              # [B, T, D]

        # ── Mean pooling (chỉ trên non-PAD tokens) ───────────────────────────
        # attention_mask: [B, T] → [B, T, 1]
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_repr   = (x * mask_expanded).sum(dim=1)          # [B, D]
        count      = mask_expanded.sum(dim=1).clamp(min=1e-9) # [B, 1]
        pooled     = sum_repr / count                          # [B, D]

        # Classification
        logits = self.classifier(pooled)   # [B, num_classes]

        return {
            "logits":       logits,
            "attn_weights": all_attn_weights,  # for visualization
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
