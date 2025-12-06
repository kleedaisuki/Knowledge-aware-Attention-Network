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

from kan.models.transformer_encoder import TransformerEncoderConfig
from kan.models.knowledge_encoder import KnowledgeEncoder, KnowledgeEncoderConfig
from kan.models.attention import (
    KnowledgeAttentionConfig,
    NewsEntityAttention,
    NewsEntityContextAttention,
)
from kan.models.pooling import masked_mean_pool, cls_pool
from kan.models.bert_text_encoder import BertTextEncoder, BertTextEncoderConfig
from kan.repr.text_embedding import TextEmbedding
from kan.repr.entity_embedding import EntityEmbedding
from kan.repr.batching import BatchEncoding
from kan.utils.logging import get_logger

logger = get_logger(__name__)


# =====================================================================
# 配置：文本编码子模块（旧版） / Text encoder config (legacy)
# =====================================================================


@dataclass
class TextEncoderConfig:
    """
    @brief 文本编码配置（旧版）：封装 TransformerEncoder 以及池化策略。
           Text encoder config (legacy): wraps TransformerEncoder and pooling strategy.
    @note
        - 新版 KAN 默认使用 BertTextEncoderConfig，经由 KANConfig.text_bert 管理；
          该配置仅为兼容旧代码而保留，实际前向不会再走 TextEmbedding+Transformer 路径。
        - The new KAN uses BertTextEncoderConfig via KANConfig.text_bert by default;
          this config is kept only for backward compatibility and is not used in
          the forward pass anymore.
    """

    encoder: TransformerEncoderConfig = field(default_factory=TransformerEncoderConfig)
    """文本 Transformer 编码器配置。Config for text Transformer encoder."""

    use_cls_pooling: bool = False
    """是否使用首 token ([CLS]) 池化，否则使用 masked mean pooling。
       Whether to use first-token ([CLS]) pooling; otherwise masked mean pooling is used.
    """


# =====================================================================
# 总体配置：文本 + 知识 + 注意力 + 分类头
# =====================================================================


@dataclass
class KANConfig:
    """
    @brief KAN 总体配置，包含文本编码、知识编码、注意力与分类头参数。
           Global configuration for KAN, including text encoder, knowledge encoder,
           attention, and classifier head.
    """

    # --- 文本编码：新版 BERT 文本编码器 ---
    text_bert: BertTextEncoderConfig = field(default_factory=BertTextEncoderConfig)
    """
    @brief 文本编码器（BERT）配置。
           Configuration for BERT-based text encoder.
    """

    # --- 文本编码：旧版 Transformer 配置（仅为向后兼容） ---
    text: Optional[TextEncoderConfig] = None
    """
    @brief 旧版文本编码配置，可选；若提供，仅用于保持配置文件结构兼容，当前实现不再使用。
           Optional legacy text encoder config; kept for backward compatibility of
           config files, but not used by the current implementation.
    """

    # --- 知识编码：实体 / 上下文 Transformer ---
    knowledge: KnowledgeEncoderConfig = field(default_factory=KnowledgeEncoderConfig)
    """知识编码器配置。Knowledge encoder configuration."""

    # --- 知识注意力 ---
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
           Knowledge attention configuration shared by N-E and N-E²C modules.
    @note
        - 实际运行时，d_model 将在 KAN.__init__ 中对齐到底层 BERT 的 hidden_size；
          这里的默认值仅用于占位。
        - At runtime, d_model will be aligned to the underlying BERT hidden_size in
          KAN.__init__; the default here is merely a placeholder.
    """

    num_classes: int = 2
    """分类类别数。Number of classes for classification."""

    final_dropout: float = 0.1
    """分类头前的 dropout 比例。Dropout probability before classifier head."""

    use_entity_contexts: bool = True
    """是否在最终特征中拼接实体上下文路径 r。
       Whether to include context path r in the final concatenated features.
    """

    zero_if_no_entities: bool = True
    """
    @brief 当没有实体输入时，是否使用零向量作为实体/上下文表示。
           When no entity inputs are given, whether to use zero vectors for entity/context
           representations.
    """

    def validate(self) -> None:
        """
        @brief 做简单配置一致性检查，例如 d_model 对齐。
               Perform basic config consistency checks, e.g., matching d_model.
        @note
            - 由于文本 now 由 BERT 负责，注意力维度与知识编码器的 d_model 需要保持一致；
              文本路径维度将在运行时再进行一次检查和日志提醒。
            - Since text is now encoded by BERT, attention.d_model should match
              knowledge.encoder.d_model; the text/BERT dimension will be validated
              at runtime.
        """
        if self.attention.d_model != self.knowledge.encoder.d_model:
            logger.warning(
                "KANConfig warning: attention.d_model (%d) != knowledge.encoder.d_model (%d). "
                "They will be reconciled in KAN.__init__, but you may want to align them "
                "explicitly in your config.",
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
        @param text_embedding 文本嵌入层，用于 token ID → 向量（当前实现不会在前向中使用）。
               TextEmbedding module mapping token IDs to vectors (not used in the
               current forward pass, kept for backward compatibility).
        @param entity_embedding 实体嵌入层，用于实体 / 上下文 ID → 向量。
               EntityEmbedding module mapping entity/context IDs to vectors.
        """
        super().__init__()
        self.config = config
        self.text_embedding = (
            text_embedding  # 保留以兼容构造协议 / kept for API compatibility
        )
        self.entity_embedding = entity_embedding

        # -------- 1) 文本编码器：BERT 路径 --------
        self.text_encoder_bert = BertTextEncoder(self.config.text_bert)
        text_dim = self.text_encoder_bert.hidden_size

        # -------- 2) 对齐知识编码器维度 d_model --------
        # 如果配置中的 knowledge.encoder.d_model 与 BERT 维度不符，则在此对齐并提示。
        if self.config.knowledge.encoder.d_model != text_dim:
            logger.info(
                "Adjusting KnowledgeEncoderConfig.d_model from %d to match BERT hidden_size=%d.",
                self.config.knowledge.encoder.d_model,
                text_dim,
            )
            self.config.knowledge.encoder.d_model = text_dim

        self.knowledge_encoder = KnowledgeEncoder(self.config.knowledge)

        # 尝试检查实体嵌入维度是否与 BERT 一致（若实体嵌入暴露了 cfg.d_model 字段）
        ent_cfg = getattr(self.entity_embedding, "cfg", None)
        if ent_cfg is not None and getattr(ent_cfg, "d_model", None) != text_dim:
            logger.warning(
                "EntityEmbedding.d_model (%s) != BERT hidden_size (%d). "
                "This may cause dimension mismatch.",
                getattr(ent_cfg, "d_model", None),
                text_dim,
            )

        # -------- 3) 对齐注意力配置的 d_model，并构造注意力模块 --------
        att_cfg = self.config.attention
        if att_cfg.d_model != text_dim:
            logger.info(
                "Rebuilding KnowledgeAttentionConfig with d_model=%d (from %d) to "
                "match BERT hidden_size.",
                text_dim,
                att_cfg.d_model,
            )
            att_cfg = KnowledgeAttentionConfig(
                d_model=text_dim,
                nhead=att_cfg.nhead,
                dropout=att_cfg.dropout,
                bias=att_cfg.bias,
                batch_first=att_cfg.batch_first,
            )
        self._attn_config = att_cfg

        self.news_entity_attn = NewsEntityAttention(att_cfg)
        self.news_entity_ctx_attn = NewsEntityContextAttention(att_cfg)

        # -------- 4) 分类头：z = [p; q; r] → logits --------
        d = text_dim
        feature_dim = d  # p
        feature_dim += d  # q
        if self.config.use_entity_contexts:
            feature_dim += d  # r

        self.final_dropout = nn.Dropout(self.config.final_dropout)
        self.classifier = nn.Linear(feature_dim, self.config.num_classes)

        logger.info(
            "Initialized KAN (BERT text encoder): d_model=%d, num_classes=%d, "
            "use_entity_contexts=%s, final_dropout=%.3f",
            d,
            self.config.num_classes,
            str(self.config.use_entity_contexts),
            self.config.final_dropout,
        )

    # -----------------------------------------------------------------
    # 内部辅助：文本编码（BERT）
    # -----------------------------------------------------------------
    def _encode_text(
        self,
        token_ids: Tensor,
        token_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        @brief 使用 BERT 编码新闻文本序列并做池化，得到全局新闻表示 p。
               Encode news token sequence with BERT and pool to get global news
               representation p.
        @param token_ids 文本 ID 张量，形状为 (B, L_t)，来自 Batcher 的 token_ids。
               Text token ID tensor of shape (B, L_t), as produced by Batcher in
               BERT mode.
        @param token_padding_mask 文本 padding 掩码，形状为 (B, L_t)，True 表示 padding。
               Padding mask for text, shape (B, L_t), where True marks padding.
        @return 新闻全局表示 p，形状为 (B, D)。
                Global news representation p of shape (B, D).
        """
        # Batcher 中的 token_padding_mask: True = padding；BERT 需要 attention_mask: 1 = valid
        if token_padding_mask is not None:
            attention_mask = (~token_padding_mask).to(dtype=torch.long)
        else:
            attention_mask = None

        # BERT 编码：返回序列表示与池化表示；我们只使用 pooled 向量作为 p
        sequence_output, pooled = self.text_encoder_bert(
            input_ids=token_ids,
            attention_mask=attention_mask,
            token_type_ids=None,
        )

        # sequence_output 目前未被使用，但保留变量名以便未来扩展（如 token-level 融合）
        _ = sequence_output

        return pooled  # (B, D)

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

        # 如果上下文关闭或未提供，则默认上下文与实体相同
        if not self.config.use_entity_contexts or ctx_emb is None:
            ctx_emb = entity_emb

        # 统一使用实体 padding_mask 作为 Transformer 的 src_key_padding_mask
        q_prime, r_prime = self.knowledge_encoder(
            entity_embeddings=entity_emb,
            context_embeddings=ctx_emb,
            padding_mask=entity_padding_mask,
        )
        return q_prime, r_prime

    # -----------------------------------------------------------------
    # 前向传播：BatchEncoding → logits（+ 可选注意力权重）
    # -----------------------------------------------------------------
    def forward(
        self,
        batch: BatchEncoding,
        return_attn_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        """
        @brief 前向计算：从 BatchEncoding 计算分类 logits。
               Forward computation: compute classification logits from BatchEncoding.
        @param batch Batcher.collate 返回的 BatchEncoding。
               BatchEncoding produced by Batcher.collate.
        @param return_attn_weights 是否返回注意力权重（字典结构），主要用于可视化与调试。
               Whether to also return attention weights (as a dict) for visualization
               and debugging.
        @return (logits, aux)：
                - logits: (B, num_classes) 的分类 logits。
                - aux:   可选的附加信息（如注意力权重），或 None。
        """
        # -------- Step 1: 文本编码，得到新闻全局表示 p --------
        p = self._encode_text(
            token_ids=batch.token_ids,
            token_padding_mask=batch.token_padding_mask,
        )

        # -------- Step 2: 知识编码，得到实体 / 上下文中间表示 q' / r' --------
        q_prime, r_prime = self._encode_knowledge(
            entity_ids=batch.entity_ids,
            context_ids=batch.context_ids,
            entity_padding_mask=batch.entity_padding_mask,
            context_padding_mask=batch.context_padding_mask,
        )

        # 若完全没有实体，按配置决定是否退化为“仅文本”模型
        if q_prime is None or batch.entity_ids is None:
            if self.config.zero_if_no_entities:
                # 使用与 p 同批大小、同维度的零向量作为 q / r
                B, D = p.shape
                q = torch.zeros(B, D, device=p.device, dtype=p.dtype)
                r = torch.zeros(B, D, device=p.device, dtype=p.dtype)
                attn_info: Dict[str, Tensor] = {}
            else:
                raise ValueError(
                    "KAN forward: entity inputs are None but zero_if_no_entities=False."
                )
        else:
            attn_info: Dict[str, Tensor] = {}

            # 2.2 N-E：News → Entities，得到加权实体表示 q
            q, ne_weights = self.news_entity_attn(
                news=p,
                entity_encoded=q_prime,
                entity_padding_mask=batch.entity_padding_mask,
                need_weights=return_attn_weights,
            )
            if return_attn_weights and ne_weights is not None:
                attn_info["news_entity"] = ne_weights

            # 2.3 N-E²C：News → (Entities, Contexts)，得到加权上下文表示 r
            if self.config.use_entity_contexts and r_prime is not None:
                r, ne2c_weights = self.news_entity_ctx_attn(
                    news=p,
                    entity_encoded=q_prime,
                    context_encoded=r_prime,
                    entity_padding_mask=batch.entity_padding_mask,
                    need_weights=return_attn_weights,
                )
                if return_attn_weights and ne2c_weights is not None:
                    attn_info["news_entity_context"] = ne2c_weights
            else:
                # 若未启用上下文，则简单用零向量占位
                B, D = p.shape
                r = torch.zeros(B, D, device=p.device, dtype=p.dtype)

        # -------- Step 3: 拼接表示 z = [p; q; r] --------
        features = [p, q]
        if self.config.use_entity_contexts and r is not None:
            features.append(r)
        z = torch.cat(features, dim=-1)

        # -------- Step 4: 分类头 --------
        z = self.final_dropout(z)
        logits = self.classifier(z)

        if return_attn_weights:
            return logits, attn_info
        return logits, None
