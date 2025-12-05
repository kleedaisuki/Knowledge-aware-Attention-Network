"""
@file entity_embedding.py
@brief 实体与实体上下文嵌入模块，为知识编码器提供输入表示。
       Entity and entity-context embedding module providing inputs for the knowledge encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from kan.utils.logging import get_logger
from kan.repr.vocab import Vocab

logger = get_logger(__name__)


# ============================================================
# 配置数据结构 Configuration
# ============================================================


@dataclass
class EntityEmbeddingConfig:
    """
    @brief 实体嵌入层配置，控制维度、dropout 等参数。
           Configuration for entity embedding layer: dimension, dropout, etc.
    @param vocab_size 实体/上下文词表大小。Vocabulary size for entities and contexts.
    @param d_model 嵌入维度，应与 KnowledgeEncoderConfig.d_model 一致。
           Embedding dimension; should match KnowledgeEncoderConfig.d_model.
    @param padding_idx PAD 实体的索引，将在嵌入表中保持为全零向量。
           Index of PAD entity, kept as zero vector in the embedding table.
    @param dropout 在嵌入之后施加的 dropout 概率。
           Dropout probability applied after embedding.
    @param share_entity_context_embeddings 若为 True，则实体与上下文共享同一嵌入表。
           If True, entities and contexts share the same embedding table.
    @param context_pooling 上下文池化方式，可选 "mean" 或 "max"。
           Pooling type for contexts, one of "mean" or "max".
    """

    vocab_size: int
    d_model: int = 128
    padding_idx: int = 0
    dropout: float = 0.1
    share_entity_context_embeddings: bool = True
    context_pooling: str = "mean"


# ============================================================
# 实体嵌入主类 Main Entity Embedding module
# ============================================================


class EntityEmbedding(nn.Module):
    """
    @brief 将实体 ID 与其上下文 ID 序列映射为向量表示，用作知识编码器输入。
           Map entity IDs and their context IDs to vector representations
           that feed into the knowledge encoder.
    """

    def __init__(self, cfg: EntityEmbeddingConfig) -> None:
        """
        @brief 根据配置构造实体与上下文嵌入层。
               Construct entity and context embedding layers from config.
        @param cfg EntityEmbeddingConfig 配置对象。Configuration object.
        """
        super().__init__()
        self.cfg = cfg

        self.entity_embedding = nn.Embedding(
            num_embeddings=cfg.vocab_size,
            embedding_dim=cfg.d_model,
            padding_idx=cfg.padding_idx,
        )

        if cfg.share_entity_context_embeddings:
            self.context_embedding = self.entity_embedding
        else:
            self.context_embedding = nn.Embedding(
                num_embeddings=cfg.vocab_size,
                embedding_dim=cfg.d_model,
                padding_idx=cfg.padding_idx,
            )

        self.dropout = nn.Dropout(cfg.dropout)

        logger.info(
            "Initialized EntityEmbedding: vocab_size=%d, d_model=%d, padding_idx=%d, "
            "dropout=%.3f, share_entity_context_embeddings=%s, context_pooling=%s",
            cfg.vocab_size,
            cfg.d_model,
            cfg.padding_idx,
            cfg.dropout,
            str(cfg.share_entity_context_embeddings),
            cfg.context_pooling,
        )

    # --------------------------------------------------------
    # 便捷构造函数 Helper constructors
    # --------------------------------------------------------
    @classmethod
    def from_vocab(
        cls,
        vocab: Vocab,
        d_model: int,
        dropout: float = 0.1,
        share_entity_context_embeddings: bool = True,
        context_pooling: str = "mean",
    ) -> "EntityEmbedding":
        """
        @brief 使用 Vocab 实例快速构造 EntityEmbedding。
               Convenience constructor to build EntityEmbedding from a Vocab instance.
        @param vocab 实体词表对象。Vocabulary object for entities / contexts.
        @param d_model 嵌入维度，应与 KnowledgeEncoder 的 d_model 对齐。
               Embedding dimension; should match d_model of the KnowledgeEncoder.
        @param dropout dropout 概率。Dropout probability.
        @param share_entity_context_embeddings 是否共享实体与上下文嵌入表。
               Whether to share embedding table for entities and contexts.
        @param context_pooling 上下文池化方式，"mean" 或 "max"。
               Context pooling strategy, "mean" or "max".
        @return EntityEmbedding 实例。Constructed EntityEmbedding instance.
        @example
            >>> ent_emb = EntityEmbedding.from_vocab(entity_vocab, d_model=128)
            >>> ent_vecs, ctx_vecs = ent_emb(entity_ids, context_ids, context_padding_mask)
        """
        pad_idx = vocab.pad_idx if vocab.pad_idx is not None else 0
        cfg = EntityEmbeddingConfig(
            vocab_size=len(vocab),
            d_model=d_model,
            padding_idx=pad_idx,
            dropout=dropout,
            share_entity_context_embeddings=share_entity_context_embeddings,
            context_pooling=context_pooling,
        )
        return cls(cfg)

    # --------------------------------------------------------
    # 内部工具：上下文池化 Context pooling helper
    # --------------------------------------------------------
    def _pool_context(
        self,
        context_emb: Tensor,
        context_padding_mask: Optional[Tensor],
    ) -> Tensor:
        """
        @brief 按实体对上下文嵌入进行池化，得到 (B, L_e, D)。
               Pool context embeddings per entity to shape (B, L_e, D).
        @param context_emb 上下文嵌入张量，形状为 (B, L_e, L_c, D)。
               Context embedding tensor of shape (B, L_e, L_c, D).
        @param context_padding_mask 上下文 padding 掩码 (B, L_e, L_c)，True 为 padding。
               Padding mask for contexts, shape (B, L_e, L_c), where True marks padding.
        @return 池化后的上下文表示，形状为 (B, L_e, D)。
                Pooled context representation of shape (B, L_e, D).
        @note 若某个实体没有任何有效上下文，则对应表示将为全零。
              If an entity has no valid context, its pooled representation will be zeros.
        """
        B, L_e, L_c, D = context_emb.shape

        if context_padding_mask is None:
            # 未提供 mask 时，只能退化为简单的无 mask 池化。
            # When no mask is provided, fall back to unmasked pooling.
            logger.warning(
                "EntityEmbedding._pool_context: context_padding_mask is None; "
                "falling back to unmasked pooling over context dimension."
            )
            if self.cfg.context_pooling == "max":
                # (B, L_e, D) over L_c
                return context_emb.max(dim=2).values
            # 默认 mean
            return context_emb.mean(dim=2)

        # 有效位置 mask_valid: True 表示有效（非 padding）
        mask_valid = ~context_padding_mask  # (B, L_e, L_c)
        mask_valid_f = mask_valid.unsqueeze(-1).to(
            context_emb.dtype
        )  # (B, L_e, L_c, 1)

        if self.cfg.context_pooling == "max":
            # 对 padding 位置置为 -inf，避免参与 max
            neg_inf = torch.finfo(context_emb.dtype).min
            masked = context_emb.masked_fill(~mask_valid.unsqueeze(-1), neg_inf)
            pooled = masked.max(dim=2).values  # (B, L_e, D)
            # 对全 padding 的情况，将 -inf 替换为 0
            pooled[~mask_valid.any(dim=2)] = 0.0
            return pooled

        # 默认使用 masked mean
        summed = (context_emb * mask_valid_f).sum(dim=2)  # (B, L_e, D)
        counts = mask_valid_f.sum(dim=2)  # (B, L_e, 1)

        # 避免除零：对 count=0 的位置保持为 0
        counts_clamped = counts.clamp(min=1.0)
        pooled = summed / counts_clamped
        
        # 把 (B, L_e, 1) 的 mask 扩展到 (B, L_e, D) 再进行填充
        zero_mask = (counts == 0).expand(-1, -1, pooled.size(-1))  # (B, L_e, D)
        pooled = pooled.masked_fill(zero_mask, 0.0)
        return pooled

    # --------------------------------------------------------
    # 前向计算 Forward
    # --------------------------------------------------------
    def forward(
        self,
        entity_ids: Optional[Tensor],
        context_ids: Optional[Tensor] = None,
        context_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """
        @brief 前向传播：实体 ID / 上下文 ID → 实体嵌入 / 上下文池化嵌入。
               Forward pass: entity IDs / context IDs → entity / context pooled embeddings.
        @param entity_ids 实体 ID 张量，形状为 (B, L_e)，可为 None。
               Entity ID tensor of shape (B, L_e), or None.
        @param context_ids 上下文 ID 张量，形状为 (B, L_e, L_c)，可为 None。
               Context ID tensor of shape (B, L_e, L_c), or None.
        @param context_padding_mask 上下文 padding 掩码 (B, L_e, L_c)，True 为 padding。
               Padding mask for contexts of shape (B, L_e, L_c), True marks padding.
        @return (entity_embeddings, context_embeddings)：
                entity_embeddings 形状为 (B, L_e, D) 或 None；
                context_embeddings 形状为 (B, L_e, D) 或 None。
                Tuple (entity_embeddings, context_embeddings):
                entity_embeddings of shape (B, L_e, D) or None;
                context_embeddings of shape (B, L_e, D) or None.
        @note 若 entity_ids 为 None，则直接返回 (None, None)。
              If entity_ids is None, returns (None, None) immediately.
        """
        if entity_ids is None:
            return None, None

        # --- 实体嵌入 Entity embeddings ---
        # (B, L_e) -> (B, L_e, D)
        entity_emb = self.entity_embedding(entity_ids)
        entity_emb = self.dropout(entity_emb)

        # --- 上下文嵌入 Context embeddings ---
        context_emb_pooled: Optional[Tensor]
        if context_ids is None:
            context_emb_pooled = None
        else:
            # (B, L_e, L_c) -> (B, L_e, L_c, D)
            context_emb = self.context_embedding(context_ids)
            # 池化到 (B, L_e, D)
            context_emb_pooled = self._pool_context(
                context_emb=context_emb,
                context_padding_mask=context_padding_mask,
            )
            context_emb_pooled = self.dropout(context_emb_pooled)

        return entity_emb, context_emb_pooled
