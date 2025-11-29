"""
@file kan.py
@brief KAN 主模型：从批次 ID 编码到最终分类 logits，内部集成嵌入、编码器与注意力。
       Main KAN model: from batched ID encoding to final classification logits,
       integrating embeddings, encoders and attentions inside.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from kan.models.transformer_encoder import (
    TransformerEncoder,
    TransformerEncoderConfig,
)
from kan.models.knowledge_encoder import (
    KnowledgeEncoder,
    KnowledgeEncoderConfig,
)
from kan.models.attention import (
    KnowledgeAttentionConfig,
    NewsEntityAttention,
    NewsEntityContextAttention,
)
from kan.models.pooling import masked_mean_pool, cls_pool
from kan.utils.logging import get_logger

# repr 子系统：词 ID → 向量
from kan.repr.text_embedding import TextEmbedding
from kan.repr.entity_embedding import EntityEmbedding
from kan.repr.batching import BatchEncoding

logger = get_logger(__name__)


# =====================================================================
# 配置：文本编码子模块
# =====================================================================


@dataclass
class TextEncoderConfig:
    """
    @brief 文本编码配置：封装 TransformerEncoder 以及池化策略。
           Text encoder config: wraps TransformerEncoder and pooling strategy.
    """

    encoder: TransformerEncoderConfig = field(default_factory=TransformerEncoderConfig)
    """文本 Transformer 编码器配置。Config for text Transformer encoder."""

    use_cls_pooling: bool = False
    """是否使用首 token ([CLS]) 池化，否则使用 masked mean pooling。
       Whether to use first-token ([CLS]) pooling; otherwise masked mean pooling is used.
    """


# =====================================================================
# 配置：整体 KAN
# =====================================================================


@dataclass
class KANConfig:
    """
    @brief KAN 总体配置，包含文本编码、知识编码、注意力与分类头参数。
           Global configuration for KAN, including text encoder, knowledge encoder,
           attention, and classifier head.
    """

    text: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    """文本编码器配置。Text encoder configuration."""

    knowledge: KnowledgeEncoderConfig = field(default_factory=KnowledgeEncoderConfig)
    """知识编码器配置。Knowledge encoder configuration."""

    attention: KnowledgeAttentionConfig = field(
        default_factory=lambda: KnowledgeAttentionConfig(
            d_model=TransformerEncoderConfig().d_model,
            nhead=TransformerEncoderConfig().nhead,
            dropout=0.1,
            bias=True,
            batch_first=True,
        )
    )
    """
    @brief 知识注意力配置（N-E 与 N-E²C 共用）。
           Config for knowledge attention (shared by N-E and N-E²C).
    @note 默认 d_model 与 TransformerEncoderConfig 保持一致（128）。
          By default, d_model is aligned with TransformerEncoderConfig (128).
    """

    num_classes: int = 2
    """分类类别数，假新闻检测中默认为 2（真 / 假）。
       Number of classes, 2 for fake-news detection.
    """

    final_dropout: float = 0.1
    """拼接后的表示在分类前的 dropout。
       Dropout probability before classifier.
    """

    use_entity_contexts: bool = True
    """是否启用实体上下文路径 N-E²C；若 False，只使用 N-E（实体）分支。
       Whether to enable entity context path (N-E²C). If False, only N-E (entities) is used.
    """

    zero_if_no_entities: bool = True
    """当没有实体输入时，是否使用全零向量代替实体 / 上下文表示。
       When no entity inputs are given, whether to use zero vectors for entity/context
       representations.
    """

    def validate(self) -> None:
        """
        @brief 做简单配置一致性检查，例如 d_model 对齐。
               Perform basic config consistency checks, e.g., matching d_model.
        @note 目前仅检查 attention.d_model 与 encoder.d_model 是否一致。
              Currently only checks attention.d_model equals text.encoder.d_model.
        """
        if self.attention.d_model != self.text.encoder.d_model:
            raise ValueError(
                "KANConfig mismatch: attention.d_model "
                f"{self.attention.d_model} != text.encoder.d_model {self.text.encoder.d_model}. "
                "Please align them explicitly."
            )
        if self.attention.d_model != self.knowledge.encoder.d_model:
            # 这里不强制 raise，而是提示；因为 KnowledgeEncoderConfig 可能被手动修改
            # We only warn here because KnowledgeEncoderConfig may be overridden manually
            logger.warning(
                "KANConfig warning: attention.d_model (%d) != knowledge.encoder.d_model (%d). "
                "Make sure entity/context embeddings have the same dimension as text encoder.",
                self.attention.d_model,
                self.knowledge.encoder.d_model,
            )


# =====================================================================
# 模型本体：内置嵌入 + 编码器 + 注意力 + 分类头
# =====================================================================


class KAN(nn.Module):
    """
    @brief KAN 主网络：从 BatchEncoding（ID + mask）到标签 logits 的端到端模型。
           Main KAN network: end-to-end model from BatchEncoding (IDs + masks)
           to label logits.
    """

    def __init__(
        self,
        config: KANConfig,
        text_embedding: TextEmbedding,
        entity_embedding: EntityEmbedding,
    ) -> None:
        """
        @brief 构造 KAN 模型并注入嵌入层与子模块。
               Construct KAN model and inject embedding layers and submodules.
        @param config KANConfig 配置对象。
               KANConfig configuration instance.
        @param text_embedding 文本嵌入层，用于 token ID → 向量。
               TextEmbedding module mapping token IDs to vectors.
        @param entity_embedding 实体嵌入层，用于实体 / 上下文 ID → 向量。
               EntityEmbedding module mapping entity/context IDs to vectors.
        """
        super().__init__()
        self.config = config
        self.config.validate()

        # 嵌入层（repr 子系统）
        self.text_embedding = text_embedding
        self.entity_embedding = entity_embedding

        # 文本编码器：S -> S'
        self.text_encoder = TransformerEncoder(self.config.text.encoder)

        # 知识编码器：E, EC -> q', r'
        self.knowledge_encoder = KnowledgeEncoder(self.config.knowledge)

        # N-E / N-E²C 注意力模块
        self.news_entity_attn = NewsEntityAttention(self.config.attention)
        self.news_entity_ctx_attn = NewsEntityContextAttention(self.config.attention)

        # 分类头：z = [p; q; r] → logits
        d = self.config.attention.d_model
        feature_dim = d  # p
        feature_dim += d  # q
        if self.config.use_entity_contexts:
            feature_dim += d  # r

        self.final_dropout = nn.Dropout(self.config.final_dropout)
        self.classifier = nn.Linear(feature_dim, self.config.num_classes)

        logger.info(
            "Initialized KAN (with embeddings): d_model=%d, num_classes=%d, "
            "use_entity_contexts=%s, final_dropout=%.3f",
            d,
            self.config.num_classes,
            str(self.config.use_entity_contexts),
            self.config.final_dropout,
        )

    # -----------------------------------------------------------------
    # 内部辅助：文本编码
    # -----------------------------------------------------------------
    def _encode_text(
        self,
        token_ids: Tensor,
        token_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        @brief 编码新闻文本序列并做池化，得到全局新闻表示 p。
               Encode news token sequence and pool to get global news representation p.
        @param token_ids 文本 ID 张量，形状为 (B, L_t)。
               Text token ID tensor of shape (B, L_t).
        @param token_padding_mask 文本 padding 掩码，形状为 (B, L_t)，True 表示 padding。
               Padding mask for text, shape (B, L_t), True marks padding positions.
        @return 新闻全局表示 p，形状为 (B, D)。
                Global news representation p of shape (B, D).
        """
        # ID → 嵌入： (B, L_t) -> (B, L_t, D)
        news_embeddings = self.text_embedding(token_ids)

        # Transformer 编码：S' -> encoded(S')
        encoded = self.text_encoder(
            news_embeddings,
            src_key_padding_mask=token_padding_mask,
        )

        # 池化成 (B, D)：CLS 或 masked mean
        if self.config.text.use_cls_pooling:
            p = cls_pool(encoded)
        else:
            p = masked_mean_pool(encoded, padding_mask=token_padding_mask)
        return p

    # -----------------------------------------------------------------
    # 内部辅助：知识编码（实体 + 上下文）
    # -----------------------------------------------------------------
    def _encode_knowledge(
        self,
        entity_ids: Optional[Tensor],
        context_ids: Optional[Tensor],
        entity_padding_mask: Optional[Tensor],
        context_padding_mask: Optional[Tensor],
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """
        @brief 编码实体与实体上下文 ID，获取中间表示 q' 与 r'。
               Encode entity and entity-context IDs to obtain q' and r'.
        @param entity_ids 实体 ID 张量，形状为 (B, L_e)，可为 None。
               Entity ID tensor of shape (B, L_e), or None.
        @param context_ids 实体上下文 ID 张量，形状为 (B, L_e, L_c)，可为 None。
               Context ID tensor of shape (B, L_e, L_c), or None.
        @param entity_padding_mask 实体 padding 掩码 (B, L_e)，True 为 padding。
               Padding mask for entities, shape (B, L_e), True marks padding.
        @param context_padding_mask 实体上下文 padding 掩码 (B, L_e, L_c)，True 为 padding。
               Padding mask for context IDs, shape (B, L_e, L_c), True marks padding.
        @return (q_prime, r_prime)，若无实体则二者均为 None。
                Tuple (q_prime, r_prime); both None if no entities.
        """
        if entity_ids is None:
            return None, None

        # 实体 / 上下文 ID → 嵌入
        # entity_emb: (B, L_e, D), ctx_emb: (B, L_e, D) 或 None（取决于实现）
        entity_emb, ctx_emb = self.entity_embedding(
            entity_ids=entity_ids,
            context_ids=context_ids,
            context_padding_mask=context_padding_mask,
        )

        # 交给 KnowledgeEncoder 处理序列结构
        q_prime, r_prime = self.knowledge_encoder(
            entity_embeddings=entity_emb,
            context_embeddings=ctx_emb,
            padding_mask=entity_padding_mask,
        )
        return q_prime, r_prime

    # -----------------------------------------------------------------
    # 前向：BatchEncoding → logits (+ attention)
    # -----------------------------------------------------------------
    def forward(
        self,
        batch: BatchEncoding,
        *,
        return_attn_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        """
        @brief 前向传播：实现 BatchEncoding → logits 的完整 KAN 流水线。
               Forward pass: full KAN pipeline mapping BatchEncoding to logits.
        @param batch BatchEncoding 批次张量封装（ID + mask + 可选标签）。
               BatchEncoding instance containing IDs, masks and optional labels.
        @param return_attn_weights 若为 True，返回注意力权重以便可视化 / 解释。
               If True, also returns attention weights for analysis / visualization.
        @return (logits, aux)：
                - logits 形状为 (B, num_classes)；
                  logits of shape (B, num_classes);
                - aux 为可选字典，包含 'ne_weights' 与 'ne2c_weights'（如请求）。
                  aux is an optional dict with 'ne_weights'/'ne2c_weights' if requested.
        """
        # -------- Step 1: 文本编码 p --------
        p = self._encode_text(
            token_ids=batch.token_ids,
            token_padding_mask=batch.token_padding_mask,
        )

        aux: Dict[str, Tensor] = {}

        # -------- Step 2: 知识编码 + 注意力 q, r --------
        if batch.entity_ids is not None:
            # 2.1 ID → 嵌入 → q', r'
            q_prime, r_prime = self._encode_knowledge(
                entity_ids=batch.entity_ids,
                context_ids=batch.context_ids,
                entity_padding_mask=batch.entity_padding_mask,
                context_padding_mask=batch.context_padding_mask,
            )

            # 理论上如果存在 entity_ids，就应该得到 q_prime
            if q_prime is None:
                raise RuntimeError(
                    "KAN: entity_ids is not None, but _encode_knowledge returned None."
                )

            # 2.2 N-E：News → Entities，得到加权实体表示 q
            q, ne_weights = self.news_entity_attn(
                news=p,
                entity_encoded=q_prime,
                entity_padding_mask=batch.entity_padding_mask,
                need_weights=return_attn_weights,
            )

            # 2.3 N-E²C：News → (Entities, Contexts)，得到加权上下文表示 r
            if self.config.use_entity_contexts and r_prime is not None:
                r, ne2c_weights = self.news_entity_ctx_attn(
                    news=p,
                    entity_encoded=q_prime,
                    context_encoded=r_prime,
                    entity_padding_mask=batch.entity_padding_mask,
                    need_weights=return_attn_weights,
                )
            else:
                r, ne2c_weights = None, None

            if return_attn_weights:
                if ne_weights is not None:
                    aux["ne_weights"] = ne_weights
                if ne2c_weights is not None:
                    aux["ne2c_weights"] = ne2c_weights
        else:
            # -------- 无实体时的降级行为 --------
            d = p.size(-1)
            if self.config.zero_if_no_entities:
                q = torch.zeros_like(p)
                r = torch.zeros_like(p) if self.config.use_entity_contexts else None
            else:
                q = p
                r = p if self.config.use_entity_contexts else None

        # -------- Step 3: 拼接表示 z = [p; q; r] --------
        features = [p, q]
        if self.config.use_entity_contexts and r is not None:
            features.append(r)
        z = torch.cat(features, dim=-1)

        # -------- Step 4: 分类头 --------
        z = self.final_dropout(z)
        logits = self.classifier(z)

        if return_attn_weights:
            return logits, aux
        return logits, None
