"""
@file bert_text_encoder.py
@brief 基于预训练 BERT 的文本编码器，实现 KAN 中的新闻语义表示提取。
       BERT-based text encoder that produces news representations for KAN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
from torch import nn
from transformers import AutoModel, AutoConfig

from kan.utils.logging import get_logger  # 项目统一日志工具 / unified logging utility

logger = get_logger(__name__)


@dataclass
class BertTextEncoderConfig:
    """
    @brief 使用预训练 BERT 编码新闻文本的配置；替代传统的 Embedding+TransformerEncoder 文本路径。
           Configuration for encoding news text with a pre-trained BERT model, replacing
           the legacy Embedding+TransformerEncoder text path.
    @param pretrained_model_name_or_path 预训练模型名称或本地路径（如 "bert-base-chinese"）。
           Name or local path of the pre-trained model (e.g. "bert-base-chinese").
    @param pooling 句子级表示的池化策略："cls" 使用 [CLS] 向量；"mean" 为简单平均；
           "mean+mask" 为基于 attention_mask 的加权平均（仅统计有效 token）。
           Pooling strategy for sentence-level representation: "cls" uses [CLS] token;
           "mean" is simple average; "mean+mask" averages only over unmasked tokens.
    @param output_dropout 句子级表示输出前的 dropout 概率。
           Dropout probability applied to the pooled sentence representation.
    @param freeze_encoder 是否冻结 BERT 编码器参数（只训练上层 KAN）。
           Whether to freeze BERT encoder parameters and only train upper layers of KAN.
    @param max_length 期望的最大序列长度（用于日志和 sanity 检查，不强制裁剪）。
           Expected maximum sequence length, used for logging and sanity checks only.
    @param return_sequence 是否在 forward 中返回完整序列表示 (B, L, H)，否则只返回 pooled。
           Whether forward returns the full sequence representation (B, L, H) in addition
           to the pooled vector; if False, only pooled output is returned.
    """

    pretrained_model_name_or_path: str = "bert-base-chinese"
    pooling: Literal["cls", "mean", "mean+mask"] = "mean"
    output_dropout: float = 0.1
    freeze_encoder: bool = False
    max_length: int = 512
    return_sequence: bool = True


class BertTextEncoder(nn.Module):
    """
    @brief 基于 HuggingFace Transformers 的 BERT 文本编码器。
           BERT text encoder built on top of HuggingFace Transformers.

    @note
        - 该模块不负责分词与构造 input_ids/attention_mask，假定上游已完成；
          This module does *not* handle tokenization or ID construction; upstream
          components are expected to provide ready-to-use input_ids/attention_mask.
        - 输出的隐藏维度等于底层 BERT 的 hidden_size（通常为 768），不做额外投影，
          以便与实体/上下文编码器直接对齐。
          The output hidden size equals the underlying BERT hidden_size (typically 768)
          and no extra projection is applied, so it aligns directly with entity/context
          encoders.
    """

    def __init__(self, cfg: BertTextEncoderConfig) -> None:
        """
        @brief 初始化 BERT 文本编码器实例。
               Initialize the BERT text encoder instance.
        @param cfg BertTextEncoderConfig 配置对象。
               BertTextEncoderConfig configuration instance.
        """
        super().__init__()

        self.cfg = cfg

        # 加载预训练配置与权重 / Load pretrained config and weights
        hf_config = AutoConfig.from_pretrained(cfg.pretrained_model_name_or_path)
        self.bert = AutoModel.from_pretrained(
            cfg.pretrained_model_name_or_path, config=hf_config
        )

        self._hidden_size = hf_config.hidden_size

        # 句子级输出上的 dropout / dropout on pooled sentence representation
        self.dropout = (
            nn.Dropout(cfg.output_dropout)
            if cfg.output_dropout > 0.0
            else nn.Identity()
        )

        if cfg.freeze_encoder:
            logger.info(
                "Freezing BERT encoder parameters (no gradient updates will be applied)."
            )
            for param in self.bert.parameters():
                param.requires_grad = False

        logger.info(
            "BertTextEncoder initialized with model=%s, hidden_size=%d, pooling=%s",
            cfg.pretrained_model_name_or_path,
            self._hidden_size,
            cfg.pooling,
        )

    # ------------------------------------------------------------------
    # 公共属性 helpers / public helpers
    # ------------------------------------------------------------------

    @property
    def hidden_size(self) -> int:
        """
        @brief 返回底层 BERT 的隐藏维度大小。
               Return the hidden size of the underlying BERT encoder.
        @return 隐藏维度（通常为 768）。Hidden dimension size (usually 768).
        """
        return self._hidden_size

    @property
    def output_dim(self) -> int:
        """
        @brief 返回句子级表示的维度，与 hidden_size 相同。
               Return the dimensionality of the pooled sentence representation,
               which equals hidden_size.
        @return 句子表示维度。Dimensionality of the sentence representation.
        """
        return self._hidden_size

    # ------------------------------------------------------------------
    # 前向传播 / forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        @brief 使用 BERT 对输入的 token 序列进行编码，返回序列表示与句子级表示。
               Encode the input token sequences with BERT and return both the
               sequence representation and the pooled sentence representation.
        @param input_ids 形状为 (B, L) 的 token ID 张量。
               Token ID tensor of shape (B, L).
        @param attention_mask 形状为 (B, L) 的 mask 张量，1 表示有效 token，0 表示 padding，可为 None。
               Attention mask tensor of shape (B, L), with 1 for valid tokens and 0
               for padding; may be None.
        @param token_type_ids 形状为 (B, L) 的 segment ID 张量，用于区分句对；单句输入时可为 None。
               Segment ID tensor of shape (B, L) used to distinguish sentence pairs;
               may be None for single-sentence inputs.
        @return
            - sequence_output: (B, L, H) 维的序列表示（若 cfg.return_sequence=False，仍返回该项以保持接口统一）。
              Sequence representation of shape (B, L, H) (returned even if
              cfg.return_sequence=False to keep the interface stable).
            - pooled_output:   (B, H) 维的句子级表示，依据 cfg.pooling 规则聚合。
              Sentence-level representation of shape (B, H), aggregated according
              to cfg.pooling.
        """

        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be 2D (batch, seq_len), got shape {tuple(input_ids.shape)}"
            )

        batch_size, seq_len = input_ids.shape

        if seq_len > self.cfg.max_length:
            # 只做日志告警，不强制截断；真正的截断由上游 batching 负责。
            # Only warn; actual truncation is expected to happen upstream.
            logger.warning(
                "Input sequence length (%d) exceeds configured max_length (%d). "
                "Ensure that truncation is handled in the batching pipeline.",
                seq_len,
                self.cfg.max_length,
            )

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        # last_hidden_state: (B, L, H)
        sequence_output: torch.Tensor = outputs.last_hidden_state

        # 句子级表示：根据配置的 pooling 策略进行聚合
        # Sentence representation: aggregate according to pooling strategy.
        if self.cfg.pooling == "cls":
            # 使用 [CLS] 位置的向量，通常为第 0 个 token
            # Use [CLS] token (position 0).
            pooled = sequence_output[:, 0, :]

        elif self.cfg.pooling == "mean":
            # 对所有位置做简单平均，不考虑 padding
            # Simple mean over all positions (ignoring padding).
            pooled = sequence_output.mean(dim=1)

        elif self.cfg.pooling == "mean+mask":
            if attention_mask is None:
                raise ValueError(
                    "mean+mask pooling requires attention_mask, but got None."
                )
            # 将 mask 扩展到与隐藏状态同维度，做加权平均
            # Expand mask to match hidden dimension and compute masked mean.
            mask = attention_mask.to(dtype=sequence_output.dtype)  # (B, L)
            mask = mask.unsqueeze(-1)  # (B, L, 1)
            masked = sequence_output * mask  # (B, L, H)

            # sum over tokens
            summed = masked.sum(dim=1)  # (B, H)

            # avoid division by zero by clamping the denominator
            denom = mask.sum(dim=1)  # (B, 1)
            denom = torch.clamp(denom, min=1e-6)

            pooled = summed / denom  # (B, H)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.cfg.pooling!r}")

        pooled = self.dropout(pooled)

        # 尽管 cfg.return_sequence 可以在将来用于裁剪输出，这里仍返回 sequence_output，
        # 以确保与未来可能需要 token-level 表示的模块兼容。
        # Although cfg.return_sequence can be used to drop sequence_output in
        # the future, we still return it here to keep the API stable for modules
        # that may need token-level representations.
        return sequence_output, pooled
