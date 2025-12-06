"""
@file transformer_encoder.py
@brief 基于 PyTorch TransformerEncoder 的通用序列编码模块，支持可配置位置编码（sinusoidal / RoPE / none）。
       Generic sequence encoder based on PyTorch TransformerEncoder, with configurable positional encoding
       (sinusoidal / RoPE / none).
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
    @brief Transformer 编码器配置，统一管理超参数与位置编码方式。
           Configuration for Transformer encoder, managing hyper-parameters and positional encoding type.
    """

    d_model: int = 768
    """隐层维度 / 模型维度。Hidden size / model dimension."""

    nhead: int = 4
    """多头自注意力头数。Number of attention heads."""

    num_layers: int = 1
    """Transformer Encoder 堆叠层数。Number of encoder layers."""

    dim_feedforward: int = 2048
    """前馈网络隐藏维度。Hidden size of feed-forward network."""

    dropout: float = 0.1
    """注意力与前馈层的 dropout 比例。Dropout probability."""

    activation: str = "gelu"
    """前馈网络激活函数名称（relu / gelu）。
       Name of activation function in feed-forward network (relu / gelu)."""

    layer_norm_eps: float = 1e-5
    """LayerNorm 的 epsilon。Epsilon value for LayerNorm."""

    batch_first: bool = True
    """输入是否为 (batch, seq, feature)。If inputs are (batch, seq, feature)."""

    max_seq_len: int = 512
    """最大序列长度，用于构建位置编码。Maximum sequence length for positional encoding."""

    positional_encoding: str = "rope"
    """
    @brief 位置编码类型："sinusoidal" / "rope" / "none"。
           Positional encoding type: "sinusoidal" / "rope" / "none".
    @note
        - "sinusoidal": 使用原始正弦位置编码 + PyTorch 内置 TransformerEncoder（完全兼容旧实现）。
          "sinusoidal": use classic sine-cosine positional encoding + nn.TransformerEncoder (backward compatible).
        - "rope": 使用旋转位置编码（RoPE），内部采用自定义 EncoderLayer 实现。
          "rope": use rotary positional embedding (RoPE) with a custom encoder stack.
        - "none": 不添加任何显式位置编码，仍使用内置 TransformerEncoder。
          "none": no explicit positional encoding, still using nn.TransformerEncoder.
    """

    rope_theta: float = 10000.0
    """
    @brief RoPE 频率基数 theta，控制不同维度的角速度。通常为 10000.0。
           RoPE base theta controlling angular frequency across dimensions, usually 10000.0.
    """


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


# ============================================================
# RoPE 专用 EncoderLayer & Encoder 实现
# RoPE-specific EncoderLayer & Encoder implementation
# ============================================================


class _RoPEEncoderLayer(nn.Module):
    """
    @brief 使用旋转位置编码（RoPE）的自注意力编码层。
           Transformer encoder layer with rotary positional embedding (RoPE) applied to Q/K.
    """

    def __init__(self, cfg: TransformerEncoderConfig) -> None:
        """
        @brief 根据配置构造带 RoPE 的编码层。
               Build a RoPE-based encoder layer from configuration.
        @param cfg TransformerEncoderConfig 配置对象。
               TransformerEncoderConfig instance.
        """
        super().__init__()
        self.cfg = cfg

        self.self_attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.nhead,
            dropout=cfg.dropout,
            batch_first=cfg.batch_first,
        )

        self.linear1 = nn.Linear(cfg.d_model, cfg.dim_feedforward)
        self.linear2 = nn.Linear(cfg.dim_feedforward, cfg.d_model)

        self.dropout = nn.Dropout(cfg.dropout)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)

        self.norm1 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.norm2 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)

        if cfg.activation == "relu":
            self.activation = nn.ReLU()
        elif cfg.activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(
                f"Unsupported activation={cfg.activation!r}, expected 'relu' or 'gelu'."
            )

        if cfg.d_model % 2 != 0:
            raise ValueError(f"RoPE requires even d_model, got d_model={cfg.d_model}.")

    def _apply_rope(self, x: Tensor) -> Tensor:
        """
        @brief 对输入序列最后一维施加 RoPE 旋转编码，仅用于构造 Q/K。
               Apply rotary positional embedding (RoPE) along last dimension for Q/K.
        @param x 输入张量 (B, L, D)。Input tensor of shape (B, L, D).
        @return 施加 RoPE 后的张量 (B, L, D)。Tensor with RoPE applied, shape (B, L, D).
        """
        # 约定 batch_first=True；若为 False，可在上层统一转置后再使用。
        B, L, D = x.shape
        device = x.device
        dtype = x.dtype

        half_dim = D // 2  # 将最后一维拆成两半进行二维旋转
        x1, x2 = x[..., :half_dim], x[..., half_dim:]

        # 位置索引 [0, 1, ..., L-1]
        pos = torch.arange(L, device=device, dtype=dtype)  # (L,)

        # 不同维度的角速度：theta^{-(i / half_dim)}
        inv_freq = 1.0 / (
            self.cfg.rope_theta
            ** (torch.arange(half_dim, device=device, dtype=dtype) / half_dim)
        )  # (half_dim,)

        # 角度矩阵 angle[t, j] = t * inv_freq[j]
        angle = torch.einsum("l,d->ld", pos, inv_freq)  # (L, half_dim)
        cos = angle.cos()[None, :, :]  # (1, L, half_dim)
        sin = angle.sin()[None, :, :]  # (1, L, half_dim)

        # 旋转：(x1, x2) -> (x1', x2')
        x1_new = x1 * cos - x2 * sin
        x2_new = x1 * sin + x2 * cos

        return torch.cat([x1_new, x2_new], dim=-1)

    def forward(
        self,
        x: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        @brief 带 RoPE 的前向计算：对 Q/K 应用 RoPE，再做多头自注意力 + 前馈网络。
               Forward pass with RoPE applied to Q/K, followed by multi-head self-attention and FFN.
        @param x 输入序列，形状 (B, L, D)。Input tensor of shape (B, L, D).
        @param src_key_padding_mask padding 掩码 (B, L)，True 表示 padding。
               Padding mask of shape (B, L), True marks padding positions.
        @return 输出序列，形状同 x。Output tensor, same shape as x.
        """
        # --- Self-Attention with RoPE on Q/K ---
        residual = x

        q = self._apply_rope(x)
        k = self._apply_rope(x)
        v = x  # RoPE 只施加在 Q/K 上，V 保持原样

        attn_output, _ = self.self_attn(
            q,
            k,
            v,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        x = residual + self.dropout1(attn_output)
        x = self.norm1(x)

        # --- Feed-Forward Network ---
        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        x = residual + x
        x = self.norm2(x)

        return x


class _RoPETransformerEncoder(nn.Module):
    """
    @brief 多层堆叠的 RoPE Transformer 编码器。
           Multi-layer Transformer encoder stack with RoPE-based encoder layers.
    """

    def __init__(self, cfg: TransformerEncoderConfig) -> None:
        """
        @brief 根据配置构造 RoPE Transformer 编码器。
               Build RoPE-based Transformer encoder from configuration.
        @param cfg TransformerEncoderConfig 配置对象。Configuration instance.
        """
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(
            [_RoPEEncoderLayer(cfg) for _ in range(cfg.num_layers)]
        )
        self.norm = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)

    def forward(
        self,
        x: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        @brief 多层 RoPE 编码器前向：依次通过各层并做最终 LayerNorm。
               Forward pass through stacked RoPE encoder layers, followed by final LayerNorm.
        @param x 输入张量 (B, L, D)。Input tensor of shape (B, L, D).
        @param src_key_padding_mask padding 掩码 (B, L)，True 表示 padding。
               Padding mask of shape (B, L), True marks padding positions.
        @return 编码后的张量 (B, L, D)。Encoded tensor of shape (B, L, D).
        """
        output = x
        for layer in self.layers:
            output = layer(output, src_key_padding_mask=src_key_padding_mask)
        output = self.norm(output)
        return output


# ============================================================
# 对外暴露的 TransformerEncoder 封装
# Public TransformerEncoder wrapper
# ============================================================


class TransformerEncoder(nn.Module):
    """
    @brief KAN 使用的通用 Transformer 编码器封装（仅做序列到序列），支持可配置位置编码。
           Generic Transformer encoder wrapper for KAN (sequence-to-sequence only),
           with configurable positional encoding.
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

        # 1) 根据 positional_encoding 决定使用哪种内部实现
        # Decide backend according to positional_encoding type.
        pe_type = (config.positional_encoding or "sinusoidal").lower()

        if pe_type not in ("sinusoidal", "rope", "none"):
            raise ValueError(
                f"Unknown positional_encoding={config.positional_encoding!r}, "
                "expected 'sinusoidal', 'rope' or 'none'."
            )

        self.pos_encoding: Optional[PositionalEncoding]

        if pe_type == "rope":
            # RoPE 模式：不使用显式位置编码模块，改用自定义 RoPE TransformerEncoder
            # RoPE mode: use custom RoPE-based encoder, no explicit positional encoding module.
            self.pos_encoding = None
            self.encoder = _RoPETransformerEncoder(config)
        else:
            # 使用 PyTorch 官方 TransformerEncoderLayer 堆叠
            # Use PyTorch's built-in TransformerEncoder.
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

            # 仅在 "sinusoidal" 模式下添加正弦位置编码；"none" 则不添加任何位置编码
            # Use sinusoidal positional encoding only when pe_type == "sinusoidal".
            if pe_type == "sinusoidal":
                self.pos_encoding = PositionalEncoding(
                    d_model=config.d_model,
                    max_len=config.max_seq_len,
                    batch_first=config.batch_first,
                    dropout=0.0,
                )
            else:
                self.pos_encoding = None

        logger.info(
            "Initialized TransformerEncoder(seq2seq): d_model=%d, nhead=%d, "
            "num_layers=%d, dim_ff=%d, dropout=%.3f, pe_type=%s",
            config.d_model,
            config.nhead,
            config.num_layers,
            config.dim_feedforward,
            config.dropout,
            pe_type,
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
        # 仅在 "sinusoidal" 模式下显式添加位置编码
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)

        encoded = self.encoder(
            x,
            src_key_padding_mask=src_key_padding_mask,
        )
        return encoded
