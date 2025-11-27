"""
@file seed.py
@brief 全局随机种子控制工具。提供 set_global_seed() 来保证训练可复现性。
       Global random seed utilities. Provides set_global_seed() for reproducibility.
"""

from __future__ import annotations

import random

import numpy as np
import torch


# ============================================================
# 工具函数：设置全局随机种子
# ============================================================


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """
    @brief 设置全局随机种子，包括 random / numpy / torch / torch.cuda。
           Set global random seed for random, numpy, torch, and torch.cuda.

    @param seed 随机种子整数。Random seed integer.
    @param deterministic 是否强制 PyTorch 进入确定性模式。
                         Whether to enforce deterministic behavior in PyTorch.

    @note
        1. deterministic=True 会导致某些算子变慢，但提高可复现性。
        2. CUDA 的确定性仍可能受到硬件 & driver 影响，不能 100% 保证跨设备一致。
        3. 如果 DataLoader 使用 num_workers>0，建议配合 worker_init_fn。
    """

    # ---- Python 内置 ----
    random.seed(seed)

    # ---- NumPy ----
    try:
        np.random.seed(seed)
    except Exception:
        pass

    # ---- PyTorch (CPU) ----
    torch.manual_seed(seed)

    # ---- PyTorch (GPU) ----
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # ---- cuDNN 设置 ----
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
