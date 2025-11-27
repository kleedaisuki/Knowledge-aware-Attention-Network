"""
@file text_embedding.py
@brief 文本词嵌入与位置编码模块，为 Transformer 提供输入表示。
       Text token embedding with positional encoding for Transformer inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from kan.utils.logging import get_logger
from kan.repr.vocab import Vocab

logger = get_logger(__name__)


# ============================================================
# 配置数据结构 Configuration
# ============================================================


@dataclass
class TextEmbeddingConfig:
    """
    @brief 文本嵌入层配置，控制维度、最大长度与 dropout 等参数。
           Configuration for text embedding layer: dimension, max length, dropout, etc.
    @param vocab_size 词表大小，对应 nn.Embedding 的 num_embeddings。
           Vocabulary size, mapped to num_embeddings in nn.Embedding.
    @param d_model 嵌入维度，应与 TransformerEncoder 的 d_model 一致。
           Embedding dimension; should match d_model of TransformerEncoder.
    @param padding_idx PAD token 的索引，将在嵌入层中保持为全零向量。
           Index of PAD token, kept as zero vector in the embedding table.
    @param max_len 支持的最大序列长度，用于预生成位置编码。
           Maximum supported sequence length used to pre-compute positional encodings.
    @param dropout 在嵌入+位置编码之后施加的 dropout 概率。
           Dropout probability applied after adding positional encodings.
    """

    vocab_size: int
    d_model: int = 128
    padding_idx: int = 0
    max_len: int = 512
    dropout: float = 0.1


# ============================================================
# 正弦位置编码 Sinusoidal Positional Encoding
# ============================================================


class SinusoidalPositionalEncoding(nn.Module):
    """
    @brief 正弦位置编码模块，实现与 Transformer 论文一致的不可训练位置编码。
           Sinusoidal positional encoding as in the original Transformer paper (non-trainable).
    """

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        """
        @brief 预计算 [0, max_len) 范围内的位置编码并注册为 buffer。
               Pre-compute positional encodings for positions in [0, max_len) and register as buffer.
        @param d_model 模型维度。Model dimension.
        @param max_len 最大序列长度。Maximum sequence length.
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # pe: [max_len, d_model]
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        # 每隔两个维度使用一个频率系数
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维

        # [1, max_len, d_model] 便于与 batch 维度对齐
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        @brief 将位置编码加到输入张量上。
               Add positional encodings to the input tensor.
        @param x 输入张量，形状为 (B, L, D)。
               Input tensor of shape (B, L, D).
        @return 加上位置编码后的张量，形状仍为 (B, L, D)。
                Tensor of shape (B, L, D) with positional encodings added.
        @note 若序列长度超过 max_len，将抛出 ValueError 以提醒上游截断。
              If sequence length exceeds max_len, a ValueError is raised.
        """
        if x.size(1) > self.max_len:
            raise ValueError(
                f"SinusoidalPositionalEncoding: sequence length {x.size(1)} "
                f"exceeds max_len={self.max_len}. Please increase max_len or "
                "truncate input sequences earlier."
            )
        # pe[:, :L, :] 自动 broadcast 到 batch 维
        x = x + self.pe[:, : x.size(1), :]
        return x


# ============================================================
# 文本嵌入主类 Main Text Embedding module
# ============================================================


class TextEmbedding(nn.Module):
    """
    @brief 将 token ID 序列映射为带有位置编码的向量序列，用作 Transformer 的输入。
           Map token ID sequences to embedded vectors with positional encodings
           as inputs to a Transformer encoder.
    """

    def __init__(self, cfg: TextEmbeddingConfig) -> None:
        """
        @brief 根据配置构造嵌入层与位置编码模块。
               Construct token embedding, positional encoding and dropout from config.
        @param cfg TextEmbeddingConfig 配置对象。Configuration object.
        """
        super().__init__()
        self.cfg = cfg

        self.token_embedding = nn.Embedding(
            num_embeddings=cfg.vocab_size,
            embedding_dim=cfg.d_model,
            padding_idx=cfg.padding_idx,
        )
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=cfg.d_model,
            max_len=cfg.max_len,
        )
        self.dropout = nn.Dropout(cfg.dropout)

        logger.info(
            "Initialized TextEmbedding: vocab_size=%d, d_model=%d, padding_idx=%d, "
            "max_len=%d, dropout=%.3f",
            cfg.vocab_size,
            cfg.d_model,
            cfg.padding_idx,
            cfg.max_len,
            cfg.dropout,
        )

    # --------------------------------------------------------
    # 便捷构造函数 Helper constructors
    # --------------------------------------------------------
    @classmethod
    def from_vocab(
        cls,
        vocab: Vocab,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.1,
    ) -> "TextEmbedding":
        """
        @brief 使用 Vocab 实例快速构造 TextEmbedding。
               Convenience constructor to build TextEmbedding from a Vocab instance.
        @param vocab 文本词表对象。Vocabulary object for text tokens.
        @param d_model 嵌入维度，应与后续 Transformer 的 d_model 对齐。
               Embedding dimension; should match d_model of subsequent Transformer.
        @param max_len 最大序列长度。Maximum sequence length.
        @param dropout dropout 概率。Dropout probability.
        @return TextEmbedding 实例。Constructed TextEmbedding instance.
        @example
            >>> text_emb = TextEmbedding.from_vocab(text_vocab, d_model=128)
            >>> x = text_emb(token_ids)  # token_ids: (B, L)
        """
        pad_idx = vocab.pad_idx if vocab.pad_idx is not None else 0
        cfg = TextEmbeddingConfig(
            vocab_size=len(vocab),
            d_model=d_model,
            padding_idx=pad_idx,
            max_len=max_len,
            dropout=dropout,
        )
        return cls(cfg)

    # --------------------------------------------------------
    # 前向计算 Forward
    # --------------------------------------------------------
    def forward(self, token_ids: Tensor) -> Tensor:
        """
        @brief 前向传播：ID → 嵌入 → 加位置编码 → dropout。
               Forward pass: IDs → embeddings → add positional encodings → dropout.
        @param token_ids 文本 ID 张量，形状为 (B, L)。
               Text ID tensor of shape (B, L).
        @return 带位置编码的嵌入张量，形状为 (B, L, D)。
                Embedded tensor with positional encodings of shape (B, L, D).
        """
        # [B, L] -> [B, L, D]
        x = self.token_embedding(token_ids)
        # 位置编码
        x = self.positional_encoding(x)
        # dropout
        x = self.dropout(x)
        return x
