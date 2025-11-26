"""
@file __init__.py
@brief kan.models 子系统公共接口，统一导出编码器、注意力模块、池化函数与主 KAN 模型。
       Public interface for the kan.models subsystem, exporting encoders,
       attention modules, pooling utilities, and the main KAN model.
"""

from __future__ import annotations

# ============================================================
# 编码器 Encoders
# ============================================================

from .transformer_encoder import (
    TransformerEncoderConfig,
    TransformerEncoder,
)

from .knowledge_encoder import (
    KnowledgeEncoderConfig,
    KnowledgeEncoder,
)

# ============================================================
# 注意力模块 Attention modules
# ============================================================

from .attention import (
    KnowledgeAttentionConfig,
    NewsEntityAttention,
    NewsEntityContextAttention,
)

# ============================================================
# 池化工具 Pooling utilities
# ============================================================

from .pooling import (
    masked_mean_pool,
    cls_pool,
)

# ============================================================
# 主模型 Main KAN model
# ============================================================

from .kan import (
    KANConfig,
    KAN,
)

# ============================================================
# Package-level Public API
# ============================================================

__all__ = [
    # 编码器配置与实现 Encoder configs & implementations
    "TransformerEncoderConfig",
    "TransformerEncoder",
    "KnowledgeEncoderConfig",
    "KnowledgeEncoder",
    # 注意力相关 Attention-related
    "KnowledgeAttentionConfig",
    "NewsEntityAttention",
    "NewsEntityContextAttention",
    # 池化函数 Pooling functions
    "masked_mean_pool",
    "cls_pool",
    # 主 KAN 模型 Main KAN model
    "KANConfig",
    "KAN",
]
