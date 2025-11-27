"""
@file model_wrapper.py
@brief 为 Trainer 提供包装后的 KAN 模型，只返回 logits 张量。
       Provide a wrapped KAN model for Trainer that returns only logits tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn, Tensor


@dataclass
class KANForTrainer(nn.Module):
    """
    @brief Trainer 使用的 KAN 包装模型：内部持有原始 KAN，只暴露 logits。
           Wrapper model for Trainer: holds the original KAN, exposes logits only.
    """

    kan: nn.Module  # 实际类型是 kan.KAN，但这里用 nn.Module 兼容

    def __init__(self, kan_model: nn.Module) -> None:
        """
        @brief 构造函数，注入原始 KAN 模型。
               Constructor, inject the original KAN model.
        @param kan_model 原始 KAN 模型实例。Original KAN model instance.
        """
        super().__init__()
        self.kan = kan_model

    def forward(self, **inputs: Any) -> Tensor:
        """
        @brief 前向计算，调用内部 KAN，仅返回正类 logit（一维）。
            Forward pass: call the inner KAN and return positive-class logits only.
        @return 形状为 (B,) 的 logits 张量。
                Logits tensor of shape (B,).
        """
        outputs = self.kan(**inputs)

        # 支持 KAN.forward 返回 (logits, aux) 或直接 logits
        if isinstance(outputs, tuple) and len(outputs) >= 1:
            logits = outputs[0]
        else:
            logits = outputs

        if not isinstance(logits, Tensor):
            raise RuntimeError(
                f"KANForTrainer: expected Tensor logits, got {type(logits)!r}"
            )

        # 若是二分类多通道输出，取正类 logit；若只有一通道，直接展平为 (B,)
        if logits.dim() == 2 and logits.size(-1) == 2:
            pos_logits = logits[:, 1]
        elif logits.dim() >= 1 and logits.size(-1) == 1:
            pos_logits = logits.view(-1)
        else:
            # 兜底：假设已经是一维 (B,)
            pos_logits = logits.view(-1)

        return pos_logits
