"""
@file __init__.py
@brief kan.utils 子系统公共接口：统一导出配置加载、日志工具、评估指标与随机性控制。
       Public interface for the kan.utils subsystem: exporting configuration loader,
       logging utilities, evaluation metrics, and global randomness control tools.
"""

from __future__ import annotations

# ============================================================
# 配置加载 Configuration loader
# ============================================================

from .configs import (
    load_experiment_config,
    load_config,
)

# ============================================================
# 日志工具 Logging utilities
# ============================================================

from .logging import (
    get_logger,
    info,
    warn,
    error,
)

# ============================================================
# 评估指标 Metrics utilities
# ============================================================

from .metrics import (
    BinaryClassificationMetrics,
    compute_binary_classification_metrics,
    log_metrics,
    write_probability_csv,
)

# ============================================================
# 随机种子控制 Randomness control
# ============================================================

from .seed import (
    set_global_seed,
)

# ============================================================
# Package-level Public API
# ============================================================

__all__ = [
    # 配置
    "load_config",
    "load_experiment_config",

    # 日志
    "get_logger",
    "info",
    "warn",
    "error",

    # 度量指标
    "BinaryClassificationMetrics",
    "compute_binary_classification_metrics",
    "log_metrics",
    "write_probability_csv",

    # 随机性
    "set_global_seed",
]
