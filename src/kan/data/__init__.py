"""
@file __init__.py
@brief kan.data 子系统公共接口，统一导出数据加载、预处理、知识图谱组件。
       Public interface for the kan.data subsystem, exporting dataset loader,
       preprocessing pipeline, and knowledge graph client.
"""

from __future__ import annotations

# ============================================================
# 导出数据样本结构 Data structures
# ============================================================

from .datasets import (
    NewsSample,
    DatasetConfig,
    NewsDataset,
)

from .preprocessing import (
    PreprocessConfig,
    PreprocessedSample,
    Preprocessor,
)

from .knowledge_graph import (
    KnowledgeGraphConfig,
    KnowledgeGraphClient,
)

# ============================================================
# Package-level Public API
# ============================================================

__all__ = [
    # 数据结构
    "NewsSample",
    "PreprocessedSample",
    # 配置类
    "DatasetConfig",
    "PreprocessConfig",
    "KnowledgeGraphConfig",
    # 核心处理组件
    "NewsDataset",
    "Preprocessor",
    "KnowledgeGraphClient",
]
