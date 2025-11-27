"""
@file batching.py
@brief 批处理模块：将 PreprocessedSample 批量转换为 ID 张量与 mask，供后续嵌入与 KAN 模型使用。
       Batching module: convert PreprocessedSample into batched ID tensors and masks
       for downstream embedding layers and the KAN model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
from torch import Tensor

from kan.utils.logging import get_logger
from kan.repr.vocab import Vocab
from kan.data.preprocessing import PreprocessedSample

logger = get_logger(__name__)


# ============================================================
# 配置与中间数据结构  Configuration & Intermediate Structures
# ============================================================


@dataclass
class BatchingConfig:
    """
    @brief 批处理配置，控制是否添加 BOS/EOS、实体/上下文截断长度等。
           Batching configuration controlling BOS/EOS usage and truncation limits.
    @param add_bos_eos_text 是否为文本序列添加 BOS/EOS（若 vocab 配置了）。Whether to add BOS/EOS for text if configured in vocab.
    @param max_text_len 文本 token 最长长度，0 表示不在此阶段截断。Max text length (0 means no extra truncation here).
    @param max_entities 每条样本最多保留多少个实体，0 表示不限制。Max entities per sample (0 = unlimited).
    @param max_context_per_entity 每个实体最多保留多少个上下文实体，0 表示不限制。Max context entities per entity (0 = unlimited).
    """

    add_bos_eos_text: bool = False
    max_text_len: int = 0
    max_entities: int = 0
    max_context_per_entity: int = 0


@dataclass
class EncodedSample:
    """
    @brief 单条样本在词表编码后的中间形式（仍是 Python list，尚未 pad）。
           Intermediate representation of a single sample after vocab encoding
           (still Python lists, not padded yet).
    @param id 样本 ID。Sample ID.
    @param token_ids 文本 token 的 ID 序列。ID sequence for text tokens.
    @param entity_ids 实体 ID 序列。ID sequence for entities.
    @param context_ids 实体上下文 ID 序列，按实体对齐：List[实体索引][上下文 ID]。
           Context entity ID sequences aligned with entities: List[entity_index][context IDs].
    @param label 可选标签。Optional label.
    """

    id: int
    token_ids: List[int]
    entity_ids: List[int]
    context_ids: List[List[int]]
    label: Optional[int] = None


@dataclass
class BatchEncoding:
    """
    @brief 批次张量封装，包含文本/实体/上下文的 ID 与 mask，以及样本 ID 和可选标签。
           Batched tensor container for text/entity/context IDs & masks, plus sample IDs and optional labels.
    @param token_ids 文本 ID 张量，形状 (B, L_t)。Text ID tensor of shape (B, L_t).
    @param token_padding_mask 文本 padding 掩码，形状 (B, L_t)，True 表示为 padding。
           Padding mask for text, shape (B, L_t), where True marks padding.
    @param entity_ids 实体 ID 张量，形状 (B, L_e)，如本批次完全无实体则为 None。
           Entity ID tensor of shape (B, L_e), or None if this batch has no entities at all.
    @param entity_padding_mask 实体 padding 掩码，形状 (B, L_e)，True 表示为 padding；若无实体则为 None。
           Padding mask for entities, shape (B, L_e), True marks padding; None if no entities.
    @param context_ids 实体上下文 ID 张量，形状 (B, L_e, L_c)，若无实体则为 None。
           Context ID tensor for entities, shape (B, L_e, L_c), or None if no entities.
    @param context_padding_mask 实体上下文 padding 掩码，形状 (B, L_e, L_c)，True 表示为 padding；若无实体则为 None。
           Padding mask for context IDs, shape (B, L_e, L_c), True marks padding; None if no entities.
    @param labels 标签张量 (B, )，若为测试集则为 None。Label tensor (B,) or None for test sets.
    @param ids 样本 ID 张量 (B, )。Sample ID tensor (B,).
    """

    token_ids: Tensor
    token_padding_mask: Tensor

    entity_ids: Optional[Tensor]
    entity_padding_mask: Optional[Tensor]

    context_ids: Optional[Tensor]
    context_padding_mask: Optional[Tensor]

    labels: Optional[Tensor]
    ids: Tensor


# ============================================================
# Batcher 主类  Main Batcher
# ============================================================


class Batcher:
    """
    @brief 负责将一批 PreprocessedSample 编码为可供嵌入层与 KAN 使用的张量。
           Convert a batch of PreprocessedSample into tensors usable by embedding
           layers and the KAN model.
    """

    def __init__(
        self,
        text_vocab: Vocab,
        entity_vocab: Vocab,
        cfg: Optional[BatchingConfig] = None,
    ) -> None:
        """
        @brief 初始化 Batcher，保存词表与配置。
               Initialize the Batcher with vocabularies and configuration.
        @param text_vocab 文本 token 词表。Vocabulary for text tokens.
        @param entity_vocab 实体/上下文 ID 词表。Vocabulary for entity and context IDs.
        @param cfg BatchingConfig 配置，如为 None 则使用默认值。
               BatchingConfig instance; default config will be used if None.
        """
        self.text_vocab = text_vocab
        self.entity_vocab = entity_vocab
        self.cfg = cfg or BatchingConfig()

    # ------------------------------------------------------------
    # 单样本编码：PreprocessedSample -> EncodedSample
    # Encode a single sample
    # ------------------------------------------------------------
    def encode_sample(self, sample: PreprocessedSample) -> EncodedSample:
        """
        @brief 使用文本/实体词表对单条样本进行 ID 编码，但不进行 padding。
               Encode a single preprocessed sample into ID lists without padding.
        @param sample 预处理后的样本。Preprocessed sample.
        @return EncodedSample 编码后的中间结构。Intermediate EncodedSample.
        """
        # --- 文本编码 Text encoding ---
        tokens = list(sample.tokens)
        if self.cfg.max_text_len > 0 and len(tokens) > self.cfg.max_text_len:
            tokens = tokens[: self.cfg.max_text_len]

        token_ids = self.text_vocab.encode(
            tokens,
            add_bos_eos=self.cfg.add_bos_eos_text,
        )

        # --- 实体与上下文编码 Entity & context encoding ---
        entities = list(sample.entities)
        entity_contexts = list(sample.entity_contexts)

        # 对齐保护：如果上下文长度与实体数不一致，做截断/补空
        if len(entity_contexts) < len(entities):
            entity_contexts.extend(
                [[] for _ in range(len(entities) - len(entity_contexts))]
            )
        elif len(entity_contexts) > len(entities):
            entity_contexts = entity_contexts[: len(entities)]

        # 根据配置截断实体数量
        if self.cfg.max_entities > 0 and len(entities) > self.cfg.max_entities:
            entities = entities[: self.cfg.max_entities]
            entity_contexts = entity_contexts[: self.cfg.max_entities]

        # 实体 ID 序列
        entity_ids: List[int] = self.entity_vocab.encode(entities, add_bos_eos=False)

        # 上下文：每个实体一条序列
        context_ids: List[List[int]] = []
        for ctx in entity_contexts:
            ctx_list = list(ctx)
            if (
                self.cfg.max_context_per_entity > 0
                and len(ctx_list) > self.cfg.max_context_per_entity
            ):
                ctx_list = ctx_list[: self.cfg.max_context_per_entity]
            # 可能为空，后续在 collate 时统一补 PAD
            ctx_ids = (
                self.entity_vocab.encode(ctx_list, add_bos_eos=False)
                if ctx_list
                else []
            )
            context_ids.append(ctx_ids)

        return EncodedSample(
            id=sample.id,
            token_ids=token_ids,
            entity_ids=entity_ids,
            context_ids=context_ids,
            label=sample.label,
        )

    # ------------------------------------------------------------
    # 批次拼接：List[PreprocessedSample] -> BatchEncoding
    # Collate a batch of samples
    # ------------------------------------------------------------
    def collate(self, samples: Sequence[PreprocessedSample]) -> BatchEncoding:
        """
        @brief 将一批预处理样本打包为张量形式的批次，包含 ID 与 padding mask。
               Collate a list of preprocessed samples into batched tensors with IDs and padding masks.
        @param samples 预处理样本列表。List of preprocessed samples.
        @return BatchEncoding 批次张量封装。BatchEncoding container.
        """
        if not samples:
            raise ValueError("Batcher.collate: got empty sample list.")

        encoded_samples: List[EncodedSample] = [self.encode_sample(s) for s in samples]

        # =======================
        # 文本部分 Text part
        # =======================
        text_seqs = [e.token_ids for e in encoded_samples]
        padded_text, _ = self.text_vocab.pad_batch(text_seqs)  # [B, L_t]
        token_ids = torch.tensor(padded_text, dtype=torch.long)

        # 根据 PAD ID 生成 mask（True 表示 padding）
        if self.text_vocab.pad_idx is not None:
            token_padding_mask = token_ids.eq(self.text_vocab.pad_idx)
        else:
            # 退路：无 PAD ID 时认为无 padding（不推荐，但保持鲁棒性）。
            token_padding_mask = torch.zeros_like(token_ids, dtype=torch.bool)

        # 样本 ID 与可选标签
        ids = torch.tensor([e.id for e in encoded_samples], dtype=torch.long)
        labels: Optional[Tensor]
        if encoded_samples[0].label is not None:
            labels = torch.tensor(
                [int(e.label) for e in encoded_samples], dtype=torch.long
            )
        else:
            labels = None

        # =======================
        # 实体部分 Entity part
        # =======================
        # 计算本批次的最大实体数
        max_entities = max(len(e.entity_ids) for e in encoded_samples)
        if max_entities == 0:
            # 本批次完全没有实体，后续可以用 None 触发 KAN 的 zero_if_no_entities 逻辑
            logger.debug("Batcher.collate: batch has no entities at all.")
            return BatchEncoding(
                token_ids=token_ids,
                token_padding_mask=token_padding_mask,
                entity_ids=None,
                entity_padding_mask=None,
                context_ids=None,
                context_padding_mask=None,
                labels=labels,
                ids=ids,
            )

        # 有实体的情况：先 pad 实体 ID
        entity_seqs = [e.entity_ids for e in encoded_samples]
        padded_entities, _ = self.entity_vocab.pad_batch(entity_seqs)  # [B, L_e]
        entity_ids = torch.tensor(padded_entities, dtype=torch.long)

        if self.entity_vocab.pad_idx is not None:
            entity_padding_mask = entity_ids.eq(self.entity_vocab.pad_idx)
        else:
            entity_padding_mask = torch.zeros_like(entity_ids, dtype=torch.bool)

        # =======================
        # 上下文部分 Context part
        # =======================
        # 计算本批次上下文的最大长度（每个实体的最大上下文数）
        max_ctx_len = 0
        for e in encoded_samples:
            for ctx_ids in e.context_ids:
                if len(ctx_ids) > max_ctx_len:
                    max_ctx_len = len(ctx_ids)

        # 若所有实体都没有上下文，则至少保留长度 1，以便后续 embedding 简化处理
        if max_ctx_len == 0:
            max_ctx_len = 1

        # 准备张量 [B, L_e, L_c]
        pad_val = (
            self.entity_vocab.pad_idx if self.entity_vocab.pad_idx is not None else 0
        )
        B = len(encoded_samples)
        L_e = max_entities
        L_c = max_ctx_len

        context_ids_tensor = torch.full(
            (B, L_e, L_c),
            fill_value=pad_val,
            dtype=torch.long,
        )

        context_padding_mask = torch.ones(
            (B, L_e, L_c),
            dtype=torch.bool,
        )  # 默认全 True（全 padding），再按真实内容改为 False

        for b_idx, e in enumerate(encoded_samples):
            # 逐实体填充
            for ent_idx in range(len(e.entity_ids)):
                ctx_ids = e.context_ids[ent_idx] if ent_idx < len(e.context_ids) else []
                # 截断到 max_ctx_len
                if len(ctx_ids) > L_c:
                    ctx_ids = ctx_ids[:L_c]
                # 若为空则保持 PAD，不修改（全部为 padding）
                if not ctx_ids:
                    continue
                # 写入 tensor
                context_ids_tensor[b_idx, ent_idx, : len(ctx_ids)] = torch.tensor(
                    ctx_ids, dtype=torch.long
                )
                # 对应位置标记为非 padding
                context_padding_mask[b_idx, ent_idx, : len(ctx_ids)] = False

        # 根据 PAD ID 再次修正 mask（以避免异常配置）
        if self.entity_vocab.pad_idx is not None:
            context_padding_mask = context_ids_tensor.eq(self.entity_vocab.pad_idx)

        return BatchEncoding(
            token_ids=token_ids,
            token_padding_mask=token_padding_mask,
            entity_ids=entity_ids,
            entity_padding_mask=entity_padding_mask,
            context_ids=context_ids_tensor,
            context_padding_mask=context_padding_mask,
            labels=labels,
            ids=ids,
        )

    # 让 Batcher 实例可直接调用，方便与 DataLoader 集成
    def __call__(self, samples: Sequence[PreprocessedSample]) -> BatchEncoding:
        """
        @brief 语法糖：使得 Batcher 实例可直接作为 collate_fn 使用。
               Syntactic sugar: allow using a Batcher instance directly as a collate_fn.
        @param samples 预处理样本列表。List of preprocessed samples.
        @return 批次编码结构。Batched encoding structure.
        """
        return self.collate(samples)
