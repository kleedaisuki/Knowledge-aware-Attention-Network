"""
@file transformer_encoder.py
@brief 基于 PyTorch TransformerEncoder 的通用序列编码模块。
       Generic sequence encoder based on PyTorch TransformerEncoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from kan.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TransformerEncoderConfig:
    """
    @brief Transformer 编码器配置，统一管理超参数。
           Configuration for Transformer encoder, managing hyper-parameters.
    """

    d_model: int = 128
    """隐层维度 / 模型维度。Hidden size / model dimension."""

    nhead: int = 4
    """多头自注意力头数。Number of attention heads."""

    num_layers: int = 1
    """Transformer Encoder 堆叠层数。Number of encoder layers."""

    dim_feedforward: int = 2048
    """前馈网络隐藏维度。Hidden size of feed-forward network."""

    dropout: float = 0.1
    """注意力与前馈层的 dropout 比例。Dropout probability."""

    activation: str = "relu"
    """前馈网络激活函数名称（relu / gelu）。
       Name of activation function in feed-forward network (relu / gelu)."""

    layer_norm_eps: float = 1e-5
    """LayerNorm 的 epsilon。Epsilon value for LayerNorm."""

    batch_first: bool = True
    """输入是否为 (batch, seq, feature)。If inputs are (batch, seq, feature)."""

    max_seq_len: int = 512
    """最大序列长度，用于构建位置编码。Maximum sequence length for positional encoding."""


class PositionalEncoding(nn.Module):
    """
    @brief 正弦位置编码模块，支持 batch_first。
           Sine-cosine positional encoding with optional batch_first support.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
        batch_first: bool = True,
        dropout: float = 0.0,
    ) -> None:
        """
        @brief 构造位置编码。
               Construct positional encoding module.
        @param d_model 特征维度。Feature dimension.
        @param max_len 最大序列长度。Maximum sequence length.
        @param batch_first 是否使用 (batch, seq, feature) 形式。
               Whether inputs are in (batch, seq, feature) format.
        @param dropout 位置编码后的 dropout 比例。
               Dropout probability applied after adding positional encoding.
        """
        super().__init__()
        self.d_model = d_model
        self.batch_first = batch_first
        self.dropout = nn.Dropout(p=dropout)

        # pe: (1, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        # 注册为 buffer，不参与梯度更新
        # Register as buffer so it's moved with the module and not trained
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        @brief 为输入张量添加位置编码。
               Add positional encoding to input tensor.
        @param x 输入序列，形状为 (B, L, D) 或 (L, B, D)。
               Input tensor of shape (B, L, D) or (L, B, D).
        @return 加上位置编码后的同形状张量。
                Tensor with positional encoding added, same shape as input.
        """
        if self.batch_first:
            # x: (B, L, D)
            B, L, D = x.shape
            if L > self.pe.size(1):
                raise ValueError(
                    f"Sequence length {L} exceeds max_len {self.pe.size(1)} "
                    "in PositionalEncoding."
                )
            x = x + self.pe[:, :L, :].to(x.dtype)
        else:
            # x: (L, B, D)
            L, B, D = x.shape
            if L > self.pe.size(1):
                raise ValueError(
                    f"Sequence length {L} exceeds max_len {self.pe.size(1)} "
                    "in PositionalEncoding."
                )
            tmp = x.transpose(0, 1)  # (B, L, D)
            tmp = tmp + self.pe[:, :L, :].to(tmp.dtype)
            x = tmp.transpose(0, 1)  # (L, B, D)

        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    @brief KAN 使用的通用 Transformer 编码器封装（仅做序列到序列）。
           Generic Transformer encoder wrapper for KAN (sequence-to-sequence only).
    """

    def __init__(self, config: TransformerEncoderConfig) -> None:
        """
        @brief 根据配置构造 Transformer Encoder。
               Build Transformer encoder from configuration.
        @param config TransformerEncoderConfig 配置对象。
               Configuration dataclass instance.
        """
        super().__init__()
        self.config = config

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=config.batch_first,
            layer_norm_eps=config.layer_norm_eps,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_layers,
        )

        self.pos_encoding = PositionalEncoding(
            d_model=config.d_model,
            max_len=config.max_seq_len,
            batch_first=config.batch_first,
            dropout=0.0,
        )

        logger.info(
            "Initialized TransformerEncoder(seq2seq): d_model=%d, nhead=%d, "
            "num_layers=%d, dim_ff=%d, dropout=%.3f",
            config.d_model,
            config.nhead,
            config.num_layers,
            config.dim_feedforward,
            config.dropout,
        )

    def forward(
        self,
        x: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        @brief 对输入序列进行 Transformer 编码，输出等长序列。
               Encode input sequence with Transformer, output sequence of same length.
        @param x 输入张量，形状为 (B, L, D)（若 batch_first=True）。
               Input tensor of shape (B, L, D) if batch_first=True.
        @param src_key_padding_mask padding 掩码，形状为 (B, L)，True 代表 padding 位置。
               Padding mask of shape (B, L), True marks padding positions.
        @return 编码后的序列张量，形状同 x。
                Encoded sequence tensor, same shape as x.
        """
        x = self.pos_encoding(x)
        encoded = self.encoder(
            x,
            src_key_padding_mask=src_key_padding_mask,
        )
        return encoded
