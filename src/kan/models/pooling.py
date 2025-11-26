"""
@file pooling.py
@brief 序列池化工具函数集合，用于从 (B, L, D) 得到 (B, D) 表示。
       Utility pooling functions to aggregate (B, L, D) sequences into (B, D) vectors.
"""

from __future__ import annotations

from typing import Optional

from torch import Tensor


def masked_mean_pool(
    x: Tensor,
    padding_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    @brief 使用可选 padding 掩码的平均池化，将序列压缩为向量。
           Mean pooling with optional padding mask, reducing sequence to vector.
    @param x 输入序列张量，形状为 (B, L, D)。
           Input sequence tensor of shape (B, L, D).
    @param padding_mask 可选 padding 掩码，形状为 (B, L)，True 表示 padding 位置。
           Optional padding mask of shape (B, L), where True marks padding positions.
    @return 池化后的表示张量，形状为 (B, D)。
            Pooled representation tensor of shape (B, D).
    @note 若未提供 padding_mask，则对所有位置做简单平均。
          If padding_mask is None, simple mean over all positions is used.
    """
    if padding_mask is None:
        # 简单平均：所有位置等权
        # Simple mean over the sequence dimension
        return x.mean(dim=1)

    # padding_mask: True 表示 padding，先取反得到有效 token
    # padding_mask: True for padding, invert to get valid tokens
    valid = ~padding_mask  # (B, L)
    valid = valid.to(x.dtype)
    x = x * valid.unsqueeze(-1)  # (B, L, D)

    # 每个样本的有效长度，避免除以 0
    # Count valid tokens per sample, avoid division by zero
    lengths = valid.sum(dim=1).clamp(min=1.0)  # (B,)

    # 按有效长度归一化
    # Normalize by valid lengths
    pooled = x.sum(dim=1) / lengths.unsqueeze(-1)  # (B, D)
    return pooled


def cls_pool(x: Tensor) -> Tensor:
    """
    @brief 使用第一个 token 作为整体序列表示（类似 [CLS]）。
           Use the first token as the sequence representation (like [CLS]).
    @param x 输入序列张量，形状为 (B, L, D)。
           Input sequence tensor of shape (B, L, D).
    @return 取首 token 后的表示张量，形状为 (B, D)。
            Tensor of shape (B, D) taking the first token.
    @note 需要上游在序列第一个位置放置专门的聚合 token（例如 [CLS]）。
          Requires upstream to place a special aggregation token (e.g. [CLS]) at position 0.
    """
    if x.dim() != 3:
        raise ValueError(f"Expected 3D tensor (B, L, D), got shape {tuple(x.shape)}")
    return x[:, 0, :]
