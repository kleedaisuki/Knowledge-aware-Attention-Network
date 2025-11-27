"""
@file __init__.py
@brief KAN 顶层公共接口：统一导出模型、训练、评估、配置、表示层、数据处理与工具函数。
       Top-level public API for the KAN project: unified export of models,
       training/evaluation, configs, representation modules, data utilities
       and core tools.
"""

from __future__ import annotations

# ============================================================
# 子系统汇总 Re-export Subsystems
# ============================================================

# ------------------------------
# Models 主模型与编码器
# ------------------------------
from .models import (  # type: ignore[F401]
    KANConfig,
    KAN,
    TransformerEncoderConfig,
    TransformerEncoder,
    KnowledgeEncoderConfig,
    KnowledgeEncoder,
    KnowledgeAttentionConfig,
    NewsEntityAttention,
    NewsEntityContextAttention,
    masked_mean_pool,
    cls_pool,
)

# ------------------------------
# Repr 表示层（词表 / 嵌入 / 批处理）
# ------------------------------
from .repr import (  # type: ignore[F401]
    # vocab
    VocabConfig,
    Vocab,
    build_text_vocab,
    build_entity_vocab,
    # text embedding
    TextEmbeddingConfig,
    TextEmbedding,
    # entity embedding
    EntityEmbeddingConfig,
    EntityEmbedding,
    # batching
    BatchingConfig,
    EncodedSample,
    BatchEncoding,
    Batcher,
)

# ------------------------------
# Data 数据处理组件
# ------------------------------
from .data import (  # type: ignore[F401]
    NewsSample,
    PreprocessedSample,
    DatasetConfig,
    PreprocessConfig,
    KnowledgeGraphConfig,
    NewsDataset,
    Preprocessor,
    KnowledgeGraphClient,
)

# ------------------------------
# Training 训练 / 评估流水线
# ------------------------------
from .training import (  # type: ignore[F401]
    TrainingConfig,
    EvaluationConfig,
    Trainer,
    Evaluator,
)

# ------------------------------
# Utils 工具模块
# ------------------------------
from .utils import (  # type: ignore[F401]
    load_config,
    load_experiment_config,
    get_logger,
    info,
    warn,
    error,
    BinaryClassificationMetrics,
    compute_binary_classification_metrics,
    log_metrics,
    write_probability_csv,
    set_global_seed,
    ExperimentConfig,
)

# ============================================================
# Package-level Public API
# ============================================================

__all__ = [
    # --- 模型 --- #
    "KANConfig",
    "KAN",
    "TransformerEncoderConfig",
    "TransformerEncoder",
    "KnowledgeEncoderConfig",
    "KnowledgeEncoder",
    "KnowledgeAttentionConfig",
    "NewsEntityAttention",
    "NewsEntityContextAttention",
    "masked_mean_pool",
    "cls_pool",
    # --- 表示层 repr --- #
    "VocabConfig",
    "Vocab",
    "build_text_vocab",
    "build_entity_vocab",
    "TextEmbeddingConfig",
    "TextEmbedding",
    "EntityEmbeddingConfig",
    "EntityEmbedding",
    "BatchingConfig",
    "EncodedSample",
    "BatchEncoding",
    "Batcher",
    # --- 数据 --- #
    "NewsSample",
    "PreprocessedSample",
    "DatasetConfig",
    "PreprocessConfig",
    "KnowledgeGraphConfig",
    "NewsDataset",
    "Preprocessor",
    "KnowledgeGraphClient",
    # --- 训练 & 评估 --- #
    "TrainingConfig",
    "EvaluationConfig",
    "Trainer",
    "Evaluator",
    # --- 工具 --- #
    "load_config",
    "load_experiment_config",
    "get_logger",
    "info",
    "warn",
    "error",
    "BinaryClassificationMetrics",
    "compute_binary_classification_metrics",
    "log_metrics",
    "write_probability_csv",
    "set_global_seed",
    "ExperimentConfig",
]
