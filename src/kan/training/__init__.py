"""
@file __init__.py
@brief kan.training 子系统公共接口：导出训练与评估组件。
       Public interface for the kan.training subsystem, exporting
       training and evaluation components.
"""

from __future__ import annotations

# ============================================================
# 导出配置与核心组件 Configs & Core Components
# ============================================================

from .trainer import (
    TrainingConfig,
    Trainer,
)

from .evaluator import (
    EvaluationConfig,
    Evaluator,
)

# ============================================================
# Package-level Public API
# ============================================================

__all__ = [
    # 配置类
    "TrainingConfig",
    "EvaluationConfig",
    # 核心流水线组件
    "Trainer",
    "Evaluator",
]
