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
        @brief 前向计算，调用内部 KAN，但只返回 logits。
               Forward pass: call inner KAN and return logits only.
        @param inputs KAN.forward 所需的关键字参数（news_embeddings 等）。
               Keyword arguments required by KAN.forward (news_embeddings, etc.).
        @return logits 张量，形状为 (B, num_classes)。
                Logits tensor of shape (B, num_classes).
        """
        outputs = self.kan(**inputs)
        # KAN.forward 返回 (logits, aux)
        if isinstance(outputs, tuple) and len(outputs) >= 1:
            logits = outputs[0]
            if not isinstance(logits, Tensor):
                raise RuntimeError(
                    f"KANForTrainer: expected first item of outputs to be Tensor, "
                    f"got {type(logits)!r}"
                )
            return logits
        if isinstance(outputs, Tensor):
            return outputs
        raise RuntimeError(
            f"KANForTrainer: unsupported KAN output type {type(outputs)!r}"
        )
