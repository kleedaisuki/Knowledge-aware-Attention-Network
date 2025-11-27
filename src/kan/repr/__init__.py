"""
@file __init__.py
@brief kan.repr 子系统公共接口，统一导出词表、嵌入、批处理等表示层组件。
       Public interface for the kan.repr subsystem, exporting vocabulary,
       embedding layers, and batching utilities.
"""

from __future__ import annotations

# ============================================================
# 词表模块 Vocabulary
# ============================================================

from .vocab import (
    VocabConfig,
    Vocab,
    build_text_vocab,
    build_entity_vocab,
)

# ============================================================
# 文本嵌入 Text Embedding
# ============================================================

from .text_embedding import (
    TextEmbeddingConfig,
    TextEmbedding,
)

# ============================================================
# 实体嵌入 Entity / Context Embedding
# ============================================================

from .entity_embedding import (
    EntityEmbeddingConfig,
    EntityEmbedding,
)

# ============================================================
# 批处理 Batching
# ============================================================

from .batching import (
    BatchingConfig,
    EncodedSample,
    BatchEncoding,
    Batcher,
)

# ============================================================
# Package-level Public API
# ============================================================

__all__ = [
    # --- Vocab ---
    "VocabConfig",
    "Vocab",
    "build_text_vocab",
    "build_entity_vocab",
    # --- Text Embedding ---
    "TextEmbeddingConfig",
    "TextEmbedding",
    # --- Entity Embedding ---
    "EntityEmbeddingConfig",
    "EntityEmbedding",
    # --- Batching ---
    "BatchingConfig",
    "EncodedSample",
    "BatchEncoding",
    "Batcher",
]
