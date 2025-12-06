"""
@file batching.py
@brief 批处理模块：将 PreprocessedSample 批量转换为 ID 张量与 mask，供后续嵌入与 KAN 模型使用。
       Batching module: convert PreprocessedSample into batched ID tensors and masks
       for downstream embedding layers and the KAN model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Literal

import torch
from torch import Tensor

from kan.utils.logging import get_logger
from kan.repr.vocab import Vocab
from kan.data.preprocessing import PreprocessedSample

logger = get_logger(__name__)

from transformers import AutoTokenizer

# ============================================================
# 配置与中间数据结构  Configuration & Intermediate Structures
# ============================================================


@dataclass
class BatchingConfig:
    """
    @brief 批处理配置，控制文本编码模式、是否添加 BOS/EOS、实体/上下文截断长度等。
           Batching configuration controlling text encoding mode, BOS/EOS usage,
           and truncation limits for entities and contexts.
    @param text_encoding 文本编码方式："vocab" 使用文本词表；"bert" 使用 BERT tokenizer。
           Text encoding mode: "vocab" uses text_vocab; "bert" uses a BERT tokenizer.
    @param add_bos_eos_text 是否为文本序列添加 BOS/EOS（仅在 text_encoding="vocab" 时生效）。
           Whether to add BOS/EOS for text (effective only when text_encoding="vocab").
    @param max_text_len 文本 token 最长长度，0 表示不在此阶段截断（vocab 模式）。
           Max text length for vocab-based encoding (0 means no extra truncation here).
    @param max_entities 每条样本最多保留多少个实体，0 表示不限制。
           Max entities per sample (0 = unlimited).
    @param max_context_per_entity 每个实体最多保留多少个上下文实体，0 表示不限制。
           Max context entities per entity (0 = unlimited).
    @param bert_model_name_or_path BERT 模型名称或路径，用于构造 tokenizer。
           Name or path of the BERT model used to construct the tokenizer.
    @param bert_max_length BERT 输入的最大序列长度，含 [CLS]/[SEP] 等特殊 token。
           Maximum sequence length for BERT inputs, including special tokens.
    @param bert_truncation BERT 截断策略："head" 截尾部，"tail" 截前部。
           Truncation strategy for BERT: "head" keeps the head, "tail" keeps the tail.
    """

    text_encoding: Literal["vocab", "bert"] = "vocab"

    add_bos_eos_text: bool = False
    max_text_len: int = 0
    max_entities: int = 0
    max_context_per_entity: int = 0

    bert_model_name_or_path: str = "bert-base-chinese"
    bert_max_length: int = 512
    bert_truncation: Literal["head", "tail"] = "head"


@dataclass
class EncodedSample:
    """
    @brief 单条样本在词表编码后的中间形式（仍是 Python list，尚未 pad）。
           Intermediate representation of a single sample after vocab encoding
           (still Python lists, not padded yet).
    @param id 样本 ID。Sample ID.
    @param token_ids 文本 token 的 ID 序列（vocab 模式下为 text_vocab ID；BERT 模式下一般不会使用）。
           ID sequence for text tokens (text_vocab IDs in vocab mode; usually unused
           in BERT mode).
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
    @param token_ids 文本 ID 张量，形状 (B, L_t)；在 vocab 模式下为 text_vocab ID，在 BERT 模式下为 BERT input_ids。
           Text ID tensor of shape (B, L_t); text_vocab IDs in vocab mode, BERT input_ids in BERT mode.
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

    @note
        - 在 text_encoding="vocab" 时，文本路径使用 text_vocab 编码，行为与旧版保持一致；
          When text_encoding="vocab", text is encoded with text_vocab and behavior
          is identical to the legacy implementation.
        - 在 text_encoding="bert" 时，文本路径使用 BERT tokenizer，token_ids 等价于
          input_ids，token_padding_mask 等价于 attention_mask==0。
          When text_encoding="bert", text is encoded with a BERT tokenizer; token_ids
          correspond to input_ids and token_padding_mask corresponds to attention_mask==0.
    """

    def __init__(
        self,
        text_vocab: Vocab,
        entity_vocab: Vocab,
        cfg: Optional[BatchingConfig] = None,
    ) -> None:
        """
        @brief 初始化 Batcher，保存词表与配置，并在需要时初始化 BERT tokenizer。
               Initialize the Batcher with vocabularies and configuration, and
               lazily initialize a BERT tokenizer if required.
        @param text_vocab 文本 token 词表（在 vocab 模式下使用，在 BERT 模式下可为占位）。
               Vocabulary for text tokens (used in vocab mode, may be a placeholder in BERT mode).
        @param entity_vocab 实体/上下文 ID 词表。Vocabulary for entity and context IDs.
        @param cfg BatchingConfig 配置，如为 None 则使用默认值。
               BatchingConfig instance; default config will be used if None.
        """
        self.text_vocab = text_vocab
        self.entity_vocab = entity_vocab
        self.cfg = cfg or BatchingConfig()

        self._bert_tokenizer = None

        if self.cfg.text_encoding == "bert":
            self._init_bert_tokenizer()

    # ------------------------------------------------------------
    # BERT tokenizer 初始化 & 文本抽取辅助函数
    # ------------------------------------------------------------

    def _init_bert_tokenizer(self) -> None:
        """
        @brief 初始化 BERT tokenizer（仅在 text_encoding="bert" 下调用）。
               Initialize the BERT tokenizer (only when text_encoding="bert").
        """
        logger.info(
            "Initializing BERT tokenizer for batching: model=%s",
            self.cfg.bert_model_name_or_path,
        )
        self._bert_tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.bert_model_name_or_path
        )

        if self._bert_tokenizer.pad_token_id is None:
            logger.warn(
                "BERT tokenizer has no pad_token_id; setting pad_token_id to eos_token_id."
            )
            self._bert_tokenizer.pad_token = self._bert_tokenizer.eos_token

    @staticmethod
    def _extract_text_for_bert(sample: PreprocessedSample) -> str:
        """
        @brief 从 PreprocessedSample 中提取适用于 BERT tokenizer 的原始文本。
               Extract raw text from PreprocessedSample for BERT tokenization.
        @param sample 预处理后的样本。Preprocessed sample.
        @return 可直接喂给 BERT tokenizer 的字符串。Text string suitable for BERT tokenizer.
        @note
            - 优先使用 sample.raw_text 或 sample.text；
              Prefer sample.raw_text or sample.text if available.
            - 否则退回到 tokens，中文默认直接拼接成一个连续字符串。
              Otherwise fall back to tokens; for Chinese we simply join tokens
              without spaces.
        """
        text = getattr(sample, "raw_text", None) or getattr(sample, "text", None)
        if text is not None:
            return text

        tokens = getattr(sample, "tokens", None)
        if tokens is None:
            raise ValueError(
                "PreprocessedSample must provide either 'raw_text'/'text' or 'tokens' "
                "for BERT-based batching."
            )

        # 对于中文数据集，通常 tokens 已经是字或词，直接拼接较为安全。
        # For Chinese datasets, tokens are usually characters or words; direct
        # concatenation is a reasonable default.
        return "".join(tokens)

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
        @note
            - 在 vocab 模式下，文本通过 text_vocab.encode 进行编码；
              In vocab mode, text is encoded via text_vocab.encode.
            - 在 BERT 模式下，文本的 token_ids 字段仍会被填充，但通常在 collate 中不会使用；
              In BERT mode, token_ids is still computed but typically ignored in
              collate in favor of BERT tokenizer outputs.
        """
        # --- 文本编码 Text encoding ---
        tokens = list(sample.tokens)
        if self.cfg.max_text_len > 0 and len(tokens) > self.cfg.max_text_len:
            tokens = tokens[: self.cfg.max_text_len]

        token_ids: List[int]
        if self.cfg.text_encoding == "vocab":
            token_ids = self.text_vocab.encode(
                tokens,
                add_bos_eos=self.cfg.add_bos_eos_text,
            )
        else:
            # BERT 模式下，这里的 token_ids 只是占位（不参与后续计算），保持结构兼容。
            # In BERT mode, token_ids here are placeholders to keep the structure
            # compatible and are not used downstream.
            token_ids = self.text_vocab.encode(
                tokens,
                add_bos_eos=False,
            )

        # --- 实体与上下文编码 Entity & context encoding ---
        entities = list(sample.entities)
        entity_contexts = list(sample.entity_contexts)

        # 对齐保护：如果上下文长度与实体数不一致，做截断/补空
        # Length alignment between entities and contexts.
        if len(entity_contexts) < len(entities):
            entity_contexts.extend(
                [[] for _ in range(len(entities) - len(entity_contexts))]
            )
        elif len(entity_contexts) > len(entities):
            entity_contexts = entity_contexts[: len(entities)]

        # 根据配置截断实体数量
        # Truncate number of entities if needed.
        if self.cfg.max_entities > 0 and len(entities) > self.cfg.max_entities:
            entities = entities[: self.cfg.max_entities]
            entity_contexts = entity_contexts[: self.cfg.max_entities]

        # 实体 ID 序列 Entity ID sequence
        entity_ids: List[int] = self.entity_vocab.encode(entities, add_bos_eos=False)

        # 上下文：每个实体一条序列 Context IDs per entity
        context_ids: List[List[int]] = []
        for ctx in entity_contexts:
            ctx_list = list(ctx)
            if (
                self.cfg.max_context_per_entity > 0
                and len(ctx_list) > self.cfg.max_context_per_entity
            ):
                ctx_list = ctx_list[: self.cfg.max_context_per_entity]
            # 可能为空，后续在 collate 时统一补 PAD
            # May be empty; padding will be handled in collate.
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
        if self.cfg.text_encoding == "vocab":
            # --- 旧模式：使用 text_vocab.pad_batch ---
            # Legacy mode: use text_vocab.pad_batch.
            text_seqs = [e.token_ids for e in encoded_samples]
            padded_text, _ = self.text_vocab.pad_batch(text_seqs)  # [B, L_t]
            token_ids = torch.tensor(padded_text, dtype=torch.long)

            # 根据 PAD ID 生成 mask（True 表示 padding）
            # Create padding mask based on pad_idx (True indicates padding).
            if self.text_vocab.pad_idx is not None:
                token_padding_mask = token_ids.eq(self.text_vocab.pad_idx)
            else:
                # 退路：无 PAD ID 时认为无 padding（不推荐，但保持鲁棒性）。
                # Fallback: assume no padding when pad_idx is None (not recommended).
                token_padding_mask = torch.zeros_like(token_ids, dtype=torch.bool)

        else:
            # --- BERT 模式：使用 BERT tokenizer ---
            # BERT mode: use BERT tokenizer to build input_ids and attention_mask.
            if self._bert_tokenizer is None:
                raise RuntimeError(
                    "Batcher.collate: BERT tokenizer is not initialized. "
                    "Check text_encoding and _init_bert_tokenizer."
                )

            texts = [self._extract_text_for_bert(s) for s in samples]

            # 确定 max_length：优先使用 bert_max_length，其次 max_text_len，最后退到 512。
            max_len = self.cfg.bert_max_length or self.cfg.max_text_len or 512

            pad_id = self._bert_tokenizer.pad_token_id
            if pad_id is None:
                raise ValueError(
                    "BERT tokenizer has no pad_token_id; please set pad_token before batching."
                )

            # ------------ 1) head 截断：直接用 HF 的 truncation ------------
            if self.cfg.bert_truncation == "head":
                enc = self._bert_tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                )
                token_ids = enc["input_ids"]  # (B, L_t)
                attention_mask = enc["attention_mask"]  # (B, L_t), 1=valid, 0=pad

            # ------------ 2) tail 截断：手工保留结尾，同时保住 CLS/SEP ------------
            elif self.cfg.bert_truncation == "tail":
                # 先不截断、不 pad，只加特殊符号
                raw_enc = self._bert_tokenizer(
                    texts,
                    padding=False,
                    truncation=False,
                    add_special_tokens=True,
                    return_tensors=None,
                )
                input_ids_list = raw_enc["input_ids"]  # List[List[int]]

                cls_id = self._bert_tokenizer.cls_token_id
                sep_id = self._bert_tokenizer.sep_token_id

                if cls_id is None or sep_id is None:
                    # 严格一点：要求 CLS/SEP 存在，否则 tail 语义不成立
                    raise ValueError(
                        "BERT tokenizer must have cls_token_id and sep_token_id "
                        "to support 'tail' truncation."
                    )

                truncated_ids_list = []
                for ids in input_ids_list:
                    # ids: [CLS] + inner + [SEP] (单句场景)
                    if len(ids) <= max_len:
                        truncated_ids_list.append(ids)
                        continue

                    # 如果首尾不是 CLS/SEP，就退化为简单 tail 截断（保留最后 max_len 个 token）
                    if not (ids[0] == cls_id and ids[-1] == sep_id):
                        truncated_ids_list.append(ids[-max_len:])
                        continue

                    inner = ids[1:-1]
                    inner_keep = max_len - 2
                    if inner_keep <= 0:
                        # 退化情况：max_len 太小，保不住 inner，只保 CLS/SEP 以及若干最后 token
                        truncated_ids_list.append(ids[-max_len:])
                    else:
                        inner_tail = inner[-inner_keep:]
                        new_ids = [cls_id] + inner_tail + [sep_id]
                        truncated_ids_list.append(new_ids)

                # 现在手工做 padding
                batch_max_len = min(
                    max(len(seq) for seq in truncated_ids_list),
                    max_len,
                )
                B = len(truncated_ids_list)

                token_ids = torch.full(
                    (B, batch_max_len),
                    fill_value=pad_id,
                    dtype=torch.long,
                )
                attention_mask = torch.zeros(
                    (B, batch_max_len),
                    dtype=torch.long,
                )

                for b_idx, seq in enumerate(truncated_ids_list):
                    L = min(len(seq), batch_max_len)
                    token_ids[b_idx, :L] = torch.tensor(seq[:L], dtype=torch.long)
                    attention_mask[b_idx, :L] = 1

            # ------------ 3) 配置值非法：直接报错，不再“悄悄当 head 用” ------------
            else:
                raise ValueError(
                    f"Batcher.collate: unknown bert_truncation={self.cfg.bert_truncation!r}, "
                    "expected 'head' or 'tail'."
                )

            # 统一生成 token_padding_mask：True 表示 padding
            token_padding_mask = attention_mask.eq(0)

        # =======================
        # 实体部分 Entity part
        # =======================
        max_entities = max(len(e.entity_ids) for e in encoded_samples)
        if max_entities == 0:
            # 本批次完全没有实体，后续可以用 None 触发 KAN 的 zero_if_no_entities 逻辑
            # Batch has no entities at all; downstream can handle None accordingly.
            logger.info("Batcher.collate: batch has no entities at all.")
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
        # When entities exist: pad entity ID sequences.
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
        # Compute max context length across all entities in the batch.
        max_ctx_len = 0
        for e in encoded_samples:
            for ctx_ids in e.context_ids:
                if len(ctx_ids) > max_ctx_len:
                    max_ctx_len = len(ctx_ids)

        # 若所有实体都没有上下文，则至少保留长度 1，以便后续 embedding 简化处理
        # If no entity has any context, keep length 1 for simplicity in downstream modules.
        if max_ctx_len == 0:
            max_ctx_len = 1

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
        # Default all True (all padding); then mark real positions as False.

        for b_idx, e in enumerate(encoded_samples):
            # 逐实体填充 Fill per-entity contexts.
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
        # Adjust mask based on pad_idx to avoid inconsistencies.
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
