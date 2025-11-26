"""
@file attention.py
@brief 知识注意力模块：实现 N-E 与 N-E²C 多头注意力。
       Knowledge attention modules: implement N-E and N-E²C multi-head attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from torch import Tensor, nn

from kan.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class KnowledgeAttentionConfig:
    """
    @brief 知识注意力配置，封装 Multi-Head Attention 的共有超参数。
           Configuration for knowledge attention based on Multi-Head Attention.
    """

    d_model: int
    """隐藏维度大小（embedding 维度）。Hidden size / embedding dimension."""

    nhead: int
    """注意力头数。Number of attention heads."""

    dropout: float = 0.1
    """注意力输出的 dropout 概率。Dropout probability on attention output."""

    bias: bool = True
    """是否在投影层使用偏置项。Whether to use bias in projection layers."""

    batch_first: bool = True
    """是否使用 (B, L, D) 作为输入格式。Whether to use (B, L, D) as input format."""


class NewsEntityAttention(nn.Module):
    """
    @brief N-E 注意力模块：News → Entities，多头注意力 $q = Attn(p, q', q')$。
           N-E attention: News → Entities, multi-head attention
           $q = Attn(p, q', q')$.
    """

    def __init__(self, config: KnowledgeAttentionConfig) -> None:
        """
        @brief 构造 NewsEntityAttention 模块。
               Construct NewsEntityAttention module.
        @param config KnowledgeAttentionConfig 配置对象。
               KnowledgeAttentionConfig instance.
        """
        super().__init__()
        self.config = config

        self.attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.nhead,
            dropout=config.dropout,
            bias=config.bias,
            batch_first=config.batch_first,
        )

        logger.info(
            "Initialized NewsEntityAttention: d_model=%d, nhead=%d, dropout=%.3f",
            config.d_model,
            config.nhead,
            config.dropout,
        )

    def forward(
        self,
        news: Tensor,
        entity_encoded: Tensor,
        entity_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        @brief 计算 N-E 多头注意力，得到加权后的实体表示 q。
               Compute N-E multi-head attention to obtain weighted entity
               representation q.
        @param news 新闻表示张量，可为 (B, D) 或 (B, L_q, D)。推荐使用 (B, D)。
               News representation tensor, shape (B, D) or (B, L_q, D).
               (B, D) is recommended.
        @param entity_encoded 实体中间编码 q'，形状为 (B, L_e, D)。
               Entity intermediate encoding q', shape (B, L_e, D).
        @param entity_padding_mask 实体 padding 掩码，形状为 (B, L_e)，True 为 padding。
               Entity padding mask of shape (B, L_e), True marks padding.
        @param need_weights 若为 True，则返回注意力权重矩阵。
               If True, also returns attention weights.
        @return (q, attn_weights)：
                q 的形状为：
                  * 若 news 为 (B, D)，则 q 为 (B, D)；
                  * 若 news 为 (B, L_q, D)，则 q 为 (B, L_q, D)。
                attn_weights 若 need_weights=False 则为 None，
                否则形状大致为 (B, L_q, L_e)（按头求平均后的权重）。
                Tuple of (q, attn_weights); attn_weights is None if need_weights=False.
        @note 该模块只对实体维度做加权，不做额外池化。对 news=(B, D) 的情况，
              查询长度为 1，因此输出天然是全局加权后的实体表示。
              This module only attends over entities. For news=(B, D),
              the query length is 1 so the output is already a global
              entity representation.
        """
        if news.dim() == 2:
            # (B, D) -> (B, 1, D)
            news = news.unsqueeze(1)
            squeeze_back = True
        elif news.dim() == 3:
            squeeze_back = False
        else:
            raise ValueError(
                f"news must be 2D or 3D tensor, got shape {tuple(news.shape)}"
            )

        if entity_encoded.dim() != 3:
            raise ValueError(
                f"entity_encoded must be 3D (B, L_e, D), got shape {tuple(entity_encoded.shape)}"
            )

        # MultiheadAttention: query=news, key=entity_encoded, value=entity_encoded
        attn_output, attn_weights = self.attn(
            query=news,
            key=entity_encoded,
            value=entity_encoded,
            key_padding_mask=entity_padding_mask,
            need_weights=need_weights,
        )

        if squeeze_back:
            # (B, 1, D) -> (B, D)
            attn_output = attn_output.squeeze(1)
            if attn_weights is not None:
                # attn_weights: (B, 1, L_e) -> (B, L_e)
                attn_weights = attn_weights.squeeze(1)

        if not need_weights:
            return attn_output, None
        return attn_output, attn_weights


class NewsEntityContextAttention(nn.Module):
    """
    @brief N-E²C 注意力模块：News → (Entities, Contexts)。
           使用实体作为 Key，实体上下文作为 Value：
           $r = Attn(p, q', r')$。
           N-E²C attention: News → (Entities, Contexts). Uses entities as
           keys and contexts as values: $r = Attn(p, q', r')$.
    """

    def __init__(self, config: KnowledgeAttentionConfig) -> None:
        """
        @brief 构造 NewsEntityContextAttention 模块。
               Construct NewsEntityContextAttention module.
        @param config KnowledgeAttentionConfig 配置对象。
               KnowledgeAttentionConfig instance.
        """
        super().__init__()
        self.config = config

        self.attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.nhead,
            dropout=config.dropout,
            bias=config.bias,
            batch_first=config.batch_first,
        )

        logger.info(
            "Initialized NewsEntityContextAttention: d_model=%d, nhead=%d, dropout=%.3f",
            config.d_model,
            config.nhead,
            config.dropout,
        )

    def forward(
        self,
        news: Tensor,
        entity_encoded: Tensor,
        context_encoded: Tensor,
        entity_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        @brief 计算 N-E²C 多头注意力，得到加权后的上下文表示 r。
               Compute N-E²C multi-head attention to obtain weighted context
               representation r.
        @param news 新闻表示张量，可为 (B, D) 或 (B, L_q, D)，推荐 (B, D)。
               News representation tensor, shape (B, D) or (B, L_q, D).
        @param entity_encoded 实体中间编码 q'，形状为 (B, L_e, D)，作为 Key。
               Entity intermediate encoding q', shape (B, L_e, D), used as keys.
        @param context_encoded 上下文中间编码 r'，形状为 (B, L_e, D)，作为 Value。
               Context intermediate encoding r', shape (B, L_e, D), used as values.
        @param entity_padding_mask 实体 padding 掩码，形状为 (B, L_e)，True 为 padding。
               Entity padding mask of shape (B, L_e), True marks padding.
        @param need_weights 若为 True，则返回注意力权重矩阵（与实体对齐）。
               If True, returns attention weights aligned with entities.
        @return (r, attn_weights)：
                r 的形状与 N-E 模块返回 q 的规则相同：
                  * news 为 (B, D) → r 为 (B, D)；
                  * news 为 (B, L_q, D) → r 为 (B, L_q, D)。
                attn_weights 若 need_weights=False 则为 None，
                否则形状大致为 (B, L_q, L_e)。
        @note 这里使用实体编码 q' 作为 Key，只用来“决定权重分布”，而用上下文 r'
              作为 Value，被加权求和，体现“通过实体选择上下文”的设计。
              Entities q' are used as keys to determine attention weights, while
              contexts r' are used as values to be aggregated.
        """
        if news.dim() == 2:
            news = news.unsqueeze(1)
            squeeze_back = True
        elif news.dim() == 3:
            squeeze_back = False
        else:
            raise ValueError(
                f"news must be 2D or 3D tensor, got shape {tuple(news.shape)}"
            )

        if entity_encoded.dim() != 3:
            raise ValueError(
                f"entity_encoded must be 3D (B, L_e, D), got shape {tuple(entity_encoded.shape)}"
            )

        if context_encoded.dim() != 3:
            raise ValueError(
                f"context_encoded must be 3D (B, L_e, D), got shape {tuple(context_encoded.shape)}"
            )

        if entity_encoded.shape != context_encoded.shape:
            raise ValueError(
                "entity_encoded and context_encoded must have the same shape "
                f"(B, L_e, D). Got {tuple(entity_encoded.shape)} vs {tuple(context_encoded.shape)}"
            )

        attn_output, attn_weights = self.attn(
            query=news,
            key=entity_encoded,
            value=context_encoded,
            key_padding_mask=entity_padding_mask,
            need_weights=need_weights,
        )

        if squeeze_back:
            attn_output = attn_output.squeeze(1)
            if attn_weights is not None:
                attn_weights = attn_weights.squeeze(1)

        if not need_weights:
            return attn_output, None
        return attn_output, attn_weights
