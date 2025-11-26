"""
@file datasets.py
@brief CSV 数据加载与批处理模块。提供统一的数据读取、解析、批次生成能力。
       CSV-based dataset loader with unified parsing and batching utilities.
"""

from __future__ import annotations

import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Iterator, Tuple

from kan.utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================
# 数据样本结构 Data Structure
# ============================================================


@dataclass
class NewsSample:
    """
    @brief 单条新闻样本结构。Represents one news sample.
    @param id 样本唯一编号。Unique sample ID.
    @param text 新闻文本内容。Raw news content.
    @param label 标签（训练集有，测试集无）。Label if available (0/1).
    """

    id: int
    text: str
    label: Optional[int] = None


# ============================================================
# 数据集配置 Data Config
# ============================================================


@dataclass
class DatasetConfig:
    """
    @brief 数据集配置参数。Dataset configuration class.
    @param csv_path CSV 文件路径。Path to the CSV file.
    @param batch_size 每批大小。Batch size for iteration.
    @param shuffle 是否随机打乱。Whether to shuffle samples.
    @param text_field 文本字段名。Column name for text.
    @param id_field ID 字段名。Column name for ID.
    @param label_field 标签字段名（可选）。Label column name (optional).
    """

    csv_path: str
    batch_size: int = 16
    shuffle: bool = True
    text_field: str = "text"
    id_field: str = "id"
    label_field: Optional[str] = "label"


# ============================================================
# 数据集主类 Dataset Loader
# ============================================================


class NewsDataset:
    """
    @brief 新闻数据集加载器，从 CSV 文件中读取并解析样本。
           News dataset loader that reads & parses samples from CSV.
    """

    def __init__(self, cfg: DatasetConfig) -> None:
        """
        @brief 初始化数据集并加载全部样本。Initialize the dataset and load samples.
        @param cfg DatasetConfig 配置对象。Dataset configuration object.
        """
        self.cfg = cfg
        self.samples: List[NewsSample] = []

        logger.info(f"Loading dataset from: {cfg.csv_path}")
        self._load()

    # ------------------------------------------------------------
    # 内部函数：CSV 读取与样本构建
    # ------------------------------------------------------------
    def _load(self) -> None:
        """@brief 读取 CSV 文件并构建 NewsSample 列表。
        Load CSV file and build list of NewsSample."""

        df = pd.read_csv(self.cfg.csv_path)

        for _, row in df.iterrows():
            sample = NewsSample(
                id=int(row[self.cfg.id_field]),
                text=str(row[self.cfg.text_field]),
                label=(
                    int(row[self.cfg.label_field])
                    if self.cfg.label_field in df.columns
                    else None
                ),
            )
            self.samples.append(sample)

        logger.info(f"Dataset loaded: {len(self.samples)} samples.")

    # ------------------------------------------------------------
    # 批次生成器 batch iterator
    # ------------------------------------------------------------
    def batch_iter(self) -> Iterator[List[NewsSample]]:
        """
        @brief 迭代生成一个个 batch。Yield mini-batches of samples.
        @return List[NewsSample] 一个批次样本。A batch of samples.
        """
        idx = list(range(len(self.samples)))

        if self.cfg.shuffle:
            import random

            random.shuffle(idx)

        for start in range(0, len(idx), self.cfg.batch_size):
            end = start + self.cfg.batch_size
            batch = [self.samples[i] for i in idx[start:end]]
            yield batch

    # ------------------------------------------------------------
    # 简化接口：用于 Trainer
    # ------------------------------------------------------------
    def get_texts_and_labels(
        self, batch: List[NewsSample]
    ) -> Tuple[List[str], List[int], List[int]]:
        """
        @brief 提取文本、标签与 id。Extract text, label and IDs from a batch.
        @param batch 一批 NewsSample。Batch of samples.
        @return (texts, labels, ids)
        """
        texts = [s.text for s in batch]
        labels = [s.label for s in batch] if batch[0].label is not None else []
        ids = [s.id for s in batch]
        return texts, labels, ids
