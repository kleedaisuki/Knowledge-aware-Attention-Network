"""
@file metrics.py
@brief 二分类评估指标与结果导出工具。
       Binary classification metrics and result export utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence, Union, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from kan.utils.logging import get_logger

logger = get_logger(__name__)

ArrayLike = Union[Sequence[float], Sequence[int], np.ndarray]


@dataclass
class BinaryClassificationMetrics:
    """
    @brief 二分类任务评估指标结果容器。Container for binary classification metrics.
    @param accuracy 准确率。Accuracy.
    @param precision 精确率（正类）。Precision for positive class.
    @param recall 召回率（正类）。Recall for positive class.
    @param f1 F1 分数（正类）。F1 score for positive class.
    @param auc ROC-AUC（可能因为单类样本而为 None）。ROC-AUC, may be None if undefined.
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: Optional[float] = None

    def to_dict(self) -> dict[str, float]:
        """
        @brief 将指标转为可序列化字典，便于日志或保存。
               Convert metrics to a serializable dict for logging or saving.
        @return 映射字符串到浮点数的字典，None 会被转换为 NaN。
                Dict mapping string to float, None converted to NaN.
        """
        result: dict[str, float] = {}
        for k, v in asdict(self).items():
            result[k] = float("nan") if v is None else float(v)
        return result


def _to_numpy_1d(x: Union[ArrayLike, "np.ndarray", "object"]) -> np.ndarray:
    """
    @brief 将输入统一转换为一维 numpy 数组。
           Convert input into a 1D numpy array.
    @param x 支持 list/tuple/numpy 数组/torch.Tensor。Supports list/tuple/np.ndarray/torch.Tensor.
    @return 一维 numpy 数组。1D numpy array.
    """
    # 延迟导入 torch，避免硬依赖。
    try:
        import torch  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        torch = None  # type: ignore[assignment]

    if torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    arr = np.asarray(x)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr


def compute_binary_classification_metrics(
    y_true: Union[ArrayLike, "np.ndarray", "object"],
    y_prob: Union[ArrayLike, "np.ndarray", "object"],
    threshold: float = 0.5,
    pos_label: int = 1,
) -> BinaryClassificationMetrics:
    """
    @brief 计算二分类任务的常用指标（Accuracy/Precision/Recall/F1/AUC）。
           Compute common metrics (Accuracy/Precision/Recall/F1/AUC) for binary classification.
    @param y_true 真实标签序列（0/1）。Ground-truth labels (0/1).
    @param y_prob 预测为正类(1)的概率序列。Predicted probability of positive class (1).
    @param threshold 将概率转为离散标签的阈值。Threshold to binarize probabilities.
    @param pos_label 正类标签。Label value of the positive class.
    @return BinaryClassificationMetrics 指标结果对象。Metrics result object.
    @note 若无法计算 AUC（例如只包含单一类别），auc 字段为 None。
          If AUC cannot be computed (e.g., only one class present), auc is set to None.
    """
    y_true_arr = _to_numpy_1d(y_true)
    y_prob_arr = _to_numpy_1d(y_prob)

    if y_true_arr.shape != y_prob_arr.shape:
        raise ValueError(
            f"y_true and y_prob shape mismatch: {y_true_arr.shape} vs {y_prob_arr.shape}"
        )

    # 概率转标签
    y_pred_arr = (y_prob_arr >= float(threshold)).astype(int)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))

    # average="binary" 且 pos_label 指定为正类
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        pos_label=pos_label,
        average="binary",
        zero_division=0,
    )
    precision = float(precision)
    recall = float(recall)
    f1 = float(f1)

    # AUC 可能因为只有一个类而报错，这里容错
    auc: Optional[float]
    try:
        auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Failed to compute ROC-AUC, likely due to a single-class y_true. Error: %s",
            e,
        )
        auc = None

    return BinaryClassificationMetrics(
        accuracy=acc,
        precision=precision,
        recall=recall,
        f1=f1,
        auc=auc,
    )


def log_metrics(
    metrics: BinaryClassificationMetrics,
    prefix: str = "",
) -> None:
    """
    @brief 使用项目 logger 友好地打印指标。
           Log metrics via project logger in a friendly format.
    @param metrics 指标对象。Metrics object.
    @param prefix 可选前缀字符串，用于区分不同阶段（如 "Val"）。Optional prefix like "Val".
    """
    p = f"{prefix} " if prefix else ""
    logger.info(
        "%smetrics: accuracy=%.4f, precision=%.4f, recall=%.4f, f1=%.4f, auc=%s",
        p,
        metrics.accuracy,
        metrics.precision,
        metrics.recall,
        metrics.f1,
        "nan" if metrics.auc is None else f"{metrics.auc:.4f}",
    )


def write_probability_csv(
    ids: Union[ArrayLike, "np.ndarray", "object"],
    probs: Union[ArrayLike, "np.ndarray", "object"],
    path: str,
) -> None:
    """
    @brief 将样本 id 与预测概率保存为 CSV 文件（id, prob）。
           Save sample ids and predicted probabilities as a CSV file (id, prob).
    @param ids 样本 id 序列（与数据集中的 id 对应）。Sequence of sample ids.
    @param probs 对应样本为正类(1)的预测概率。Predicted probability of positive class (1).
    @param path 输出 CSV 路径，例如 "results.csv"。Output CSV path, e.g., "results.csv".
    @note 输出文件包含两列：id, prob，且不包含行索引。
          Output CSV contains exactly two columns: id, prob, without index.
    """
    ids_arr = _to_numpy_1d(ids)
    probs_arr = _to_numpy_1d(probs)

    if ids_arr.shape != probs_arr.shape:
        raise ValueError(
            f"ids and probs shape mismatch: {ids_arr.shape} vs {probs_arr.shape}"
        )

    df = pd.DataFrame({"id": ids_arr, "prob": probs_arr})
    df.to_csv(path, index=False)
    logger.info("Prediction probabilities saved to CSV: %s", path)
