"""
@file knowledge_encoder.py
@brief 知识编码模块：使用 Transformer 对实体与实体上下文序列分别编码。
       Knowledge encoder module: use Transformer to encode entity and entity-context sequences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from torch import Tensor, nn

from kan.models.transformer_encoder import (
    TransformerEncoder,
    TransformerEncoderConfig,
)
from kan.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class KnowledgeEncoderConfig:
    """
    @brief 知识编码器配置，封装实体/上下文共用的 Transformer 超参数。
           Configuration for knowledge encoder, wrapping shared Transformer settings
           for entities and entity contexts.
    """

    encoder: TransformerEncoderConfig = field(default_factory=TransformerEncoderConfig)
    """内部 TransformerEncoder 的配置。Config for internal TransformerEncoder."""

    share_encoder: bool = False
    """是否让实体与实体上下文共用同一组 Transformer 参数。
       Whether entities and entity contexts share the same Transformer encoder.
    """


class KnowledgeEncoder(nn.Module):
    """
    @brief 对实体序列与实体上下文序列分别进行 Transformer 编码。
           Encode entity sequences and entity-context sequences with Transformer.
    """

    def __init__(self, config: KnowledgeEncoderConfig) -> None:
        """
        @brief 根据配置构造知识编码器。
               Build knowledge encoder from configuration.
        @param config KnowledgeEncoderConfig 配置对象。
               KnowledgeEncoderConfig instance.
        """
        super().__init__()
        self.config = config

        # 实体编码器
        # Encoder for entity sequences
        self.entity_encoder = TransformerEncoder(config.encoder)

        # 上下文编码器，可以选择是否与实体编码器共享参数
        # Encoder for entity contexts; may share parameters with entity encoder
        if config.share_encoder:
            self.context_encoder = self.entity_encoder
        else:
            self.context_encoder = TransformerEncoder(config.encoder)

        logger.info(
            "Initialized KnowledgeEncoder: d_model=%d, nhead=%d, num_layers=%d, "
            "share_encoder=%s",
            config.encoder.d_model,
            config.encoder.nhead,
            config.encoder.num_layers,
            str(config.share_encoder),
        )

    def forward(
        self,
        entity_embeddings: Tensor,
        context_embeddings: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        @brief 对实体与实体上下文嵌入做编码，得到中间表示 q' 与 r'。
               Encode entity and entity-context embeddings to obtain q' and r'.
        @param entity_embeddings 实体嵌入张量，形状为 (B, L, D)。
               Entity embedding tensor of shape (B, L, D).
        @param context_embeddings 实体上下文嵌入张量，形状为 (B, L, D)，可为 None，
               若为 None，则默认与 entity_embeddings 相同。
               Entity-context embedding tensor of shape (B, L, D). If None, will
               default to entity_embeddings.
        @param padding_mask padding 掩码，形状为 (B, L)，True 表示该位置为 padding。
               Padding mask tensor of shape (B, L), where True marks padding
               positions.
        @return (entity_encoded, context_encoded)，两者形状均为 (B, L, D)。
                Tuple of (entity_encoded, context_encoded), both of shape (B, L, D).
        @note 该模块仅做“序列到序列”的编码，不做池化。
              This module is sequence-to-sequence only; no pooling is applied.
        """
        if context_embeddings is None:
            context_embeddings = entity_embeddings

        if entity_embeddings.dim() != 3:
            raise ValueError(
                f"entity_embeddings must be 3D (B, L, D), got {tuple(entity_embeddings.shape)}"
            )
        if context_embeddings.dim() != 3:
            raise ValueError(
                f"context_embeddings must be 3D (B, L, D), got {tuple(context_embeddings.shape)}"
            )

        # 简单的形状一致性检查：batch 与长度需要相同
        # Basic shape consistency check: batch size and sequence length must match
        if (
            entity_embeddings.shape[0] != context_embeddings.shape[0]
            or entity_embeddings.shape[1] != context_embeddings.shape[1]
        ):
            raise ValueError(
                "entity_embeddings and context_embeddings must have same (B, L), "
                f"got {tuple(entity_embeddings.shape)} vs {tuple(context_embeddings.shape)}"
            )

        # TransformerEncoder 内部已经处理了位置编码与 padding mask 逻辑
        # TransformerEncoder handles positional encoding and src_key_padding_mask internally
        entity_encoded = self.entity_encoder(
            entity_embeddings,
            src_key_padding_mask=padding_mask,
        )
        context_encoded = self.context_encoder(
            context_embeddings,
            src_key_padding_mask=padding_mask,
        )

        return entity_encoded, context_encoded
