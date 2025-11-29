from __future__ import annotations

"""
@file helpers.py
@brief kan_cli 与底层 kan 库之间的“胶水层”工具函数集合。
       Helper utilities that connect the kan_cli frontend with the core kan
       library (data → vocabs → batches → model → optimizers).
"""

from typing import Any, Iterable, Optional, Tuple, Sequence, TypeVar

_T = TypeVar("_T")

import json
from pathlib import Path

from torch.utils.data import Dataset, DataLoader

from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler

import kan
from kan import (
    Preprocessor,
    PreprocessConfig,
    PreprocessedSample,
    KnowledgeGraphConfig,
    KnowledgeGraphClient,
    DatasetConfig,
    build_entity_vocab,
    build_text_vocab,
    NewsDataset,
    Vocab,
    BatchingConfig,
    Batcher,
    KANConfig,
    KAN,
    NewsSample,
    ExperimentConfig,
)

from kan.repr.batching import BatchEncoding  # 仅用于类型注解与文档
from kan.repr.text_embedding import TextEmbedding, TextEmbeddingConfig
from kan.repr.entity_embedding import EntityEmbedding, EntityEmbeddingConfig

from kan import get_logger

logger = get_logger(__name__)


# ============================================================
# Phase 1：基础工具：identity_collate / 预处理 Dataset
# ============================================================


def identity_collate(batch: Sequence[_T]) -> Sequence[_T]:
    """
    @brief DataLoader 批次恒等聚合函数：保持 batch 为原始序列，不做任何拼接或转换。
           Identity collate function for DataLoader: keep the batch as the
           original sequence of samples without stacking or transformation.

    @param batch 一个样本序列，例如 List[PreprocessedSample]。
           A sequence of samples, e.g. List[PreprocessedSample].
    @return 与输入完全相同的 batch 序列。
            Exactly the same batch sequence as the input.
    @note 适用于我们希望由上层组件（如 kan.repr.Batcher）负责将
          PreprocessedSample 列表转换为张量批次的场景。
          Suitable when upper-layer components (e.g. kan.repr.Batcher) are
          responsible for converting a list of PreprocessedSample into tensor
          batches.
    """
    return batch


class PreprocessedDataset(Dataset):
    """
    @brief 预处理后样本的数据集封装，兼容 PyTorch DataLoader。
           Dataset wrapper for preprocessed samples, compatible with DataLoader.
    """

    def __init__(self, samples: list[PreprocessedSample]) -> None:
        """
        @brief 使用预处理好的样本列表初始化数据集。
               Initialize dataset with a list of preprocessed samples.
        @param samples 预处理样本列表。List of preprocessed samples.
        """
        self._samples = samples

    def __len__(self) -> int:
        """
        @brief 返回数据集中样本数量。
               Return the number of samples in the dataset.
        """
        return len(self._samples)

    def __getitem__(self, idx: int) -> PreprocessedSample:
        """
        @brief 按索引获取单条样本。Get one sample by index.
        @param idx 样本索引。Sample index.
        @return 对应的 PreprocessedSample。Corresponding PreprocessedSample.
        """
        return self._samples[idx]


# ============================================================
# Phase 2：原始 CSV → PreprocessedSample
# ============================================================


def _build_preprocessed_dataset(
    cfg_dataset: DatasetConfig,
    cfg_preprocess: PreprocessConfig,
    cfg_kg: KnowledgeGraphConfig,
) -> PreprocessedDataset:
    """
    @brief 从配置构建完整的预处理数据集：CSV → NewsSample → PreprocessedSample。
           Build a fully preprocessed dataset from configs:
           CSV → NewsSample → PreprocessedSample.

    @param cfg_dataset 数据集配置。Dataset configuration.
    @param cfg_preprocess 预处理配置。Preprocess configuration.
    @param cfg_kg 知识图谱配置。Knowledge graph configuration.
    @return PreprocessedDataset 实例。PreprocessedDataset instance.
    """
    dataset = NewsDataset(cfg_dataset)
    logger.info(
        "Building preprocessed dataset from %s (num_samples=%d)",
        cfg_dataset.csv_path,
        len(dataset.samples),
    )

    kg_client: Optional[KnowledgeGraphClient] = None
    if cfg_preprocess.enable_kg:
        kg_client = KnowledgeGraphClient(cfg_kg)

    preprocessor = Preprocessor(cfg_preprocess, kg_client=kg_client)

    # 这里直接用 dataset.samples 作为 Iterable[NewsSample]
    preprocessed: list[PreprocessedSample] = preprocessor.preprocess_batch(
        dataset.samples
    )
    logger.info("Preprocessed dataset built: %d samples.", len(preprocessed))

    return PreprocessedDataset(preprocessed)


def build_all_dataloaders(
    config: ExperimentConfig,
) -> Tuple[DataLoader[Sequence[PreprocessedSample]], Optional[Any], Optional[Any]]:
    """
    @brief 构建 (train, val, test) DataLoader（预处理级别）。
           Build (train, val, test) DataLoaders at the PreprocessedSample level.
    @param config 实验配置，需至少包含 dataset / preprocess / kg 字段。
           Experiment config, expected to have dataset / preprocess / kg attributes.
    @return (train_loader, val_loader, test_loader) 三元组，当前仅构建 train。
            Triple (train_loader, val_loader, test_loader); currently only train.
    @note 输出的 train_loader 元素类型为 Sequence[PreprocessedSample]，
          需要配合 Batcher 才能得到 BatchEncoding。
          The returned train_loader yields Sequence[PreprocessedSample], which
          must be further processed by Batcher to get BatchEncoding.
    """
    ds_cfg = config.dataset
    pp_cfg = config.preprocess
    kg_cfg = config.kg

    preprocessed_dataset = _build_preprocessed_dataset(ds_cfg, pp_cfg, kg_cfg)

    train_loader: DataLoader[Sequence[PreprocessedSample]] = DataLoader(
        preprocessed_dataset,
        batch_size=ds_cfg.batch_size,
        shuffle=ds_cfg.shuffle,
        collate_fn=identity_collate,
    )

    val_loader: Optional[DataLoader] = None
    test_loader: Optional[DataLoader] = None

    return train_loader, val_loader, test_loader


# ============================================================
# Phase 3：PreprocessedSample 流 → 构建 / 加载 Vocab
# ============================================================


def _iter_preprocessed_samples(
    data: Iterable[Any],
) -> Iterable[PreprocessedSample]:
    """
    @brief 辅助函数：从任意可迭代结构中抽取 PreprocessedSample。
           Helper to extract PreprocessedSample objects from a generic iterable.
    @param data 可迭代对象，元素可以是 PreprocessedSample、其批次，或 Dataset 本身。
           Iterable whose elements can be PreprocessedSample, batches thereof,
           or a Dataset yielding PreprocessedSample.
    @return 逐个产出 PreprocessedSample 的生成器。Generator yielding PreprocessedSample.
    @throws TypeError 如元素类型不符合预期。
            Raises TypeError if element types are unsupported.
    """
    # 1) 如果是 Dataset（实现了 __len__ / __getitem__），我们直接遍历索引
    if isinstance(data, Dataset):
        for i in range(len(data)):
            sample = data[i]
            if isinstance(sample, PreprocessedSample):
                yield sample
            else:
                raise TypeError(
                    "Dataset should yield PreprocessedSample, got " f"{type(sample)!r}."
                )
        return

    # 2) 否则，就当成一般可迭代对象来处理
    for item in data:
        if isinstance(item, PreprocessedSample):
            # 单个样本
            yield item
        elif isinstance(item, Sequence):
            # 一个 batch（例如 DataLoader 的一个输出）
            for sub in item:
                if isinstance(sub, PreprocessedSample):
                    yield sub
                else:
                    raise TypeError(
                        "Unsupported element type inside batch when iterating "
                        f"PreprocessedSample: {type(sub)!r}"
                    )
        else:
            raise TypeError(
                "build_or_load_vocabs expects an iterable of PreprocessedSample "
                "or batches thereof; got "
                f"{type(item)!r}. You may need to adjust your pipeline."
            )


def build_or_load_vocabs(
    config: Any,
    work_dir: Path | str,
    train_loader: Iterable[Any],
    build_if_missing: bool = True,
) -> Tuple[Vocab, Vocab]:
    """
    @brief 构建或加载文本/实体词表，统一由 runtime 调用。
           Build or load text/entity vocabularies, used by the runtime.
    @param config 实验配置对象，需包含 text_vocab / entity_vocab 字段。
           Experiment config expected to have text_vocab / entity_vocab fields.
    @param work_dir 实验工作目录，用于存放 vocabs/ 子目录。
           Working directory where vocabs/ subdirectory will be created.
    @param train_loader 训练数据迭代器，可以是 DataLoader、Dataset 或任意 Iterable。
           Training data iterable, may be a DataLoader, Dataset, or generic iterable.
    @param build_if_missing 若为 True，在无现有词表时从训练数据构建新词表。
           If True, build new vocabs from training data when files are missing.
    @return (text_vocab, entity_vocab) 二元组。
            Pair (text_vocab, entity_vocab).
    """
    work_dir = Path(work_dir)
    vocabs_dir = work_dir / "vocabs"
    vocabs_dir.mkdir(parents=True, exist_ok=True)

    text_path = vocabs_dir / "text_vocab.json"
    entity_path = vocabs_dir / "entity_vocab.json"

    # ---------- 情况 1：磁盘上已有词表，直接加载 ----------
    if text_path.is_file() and entity_path.is_file():
        logger.info(
            "Loading vocabs from %s and %s",
            text_path.as_posix(),
            entity_path.as_posix(),
        )
        with text_path.open("r", encoding="utf-8") as f:
            text_data = json.load(f)
        with entity_path.open("r", encoding="utf-8") as f:
            entity_data = json.load(f)
        text_vocab = Vocab.from_dict(text_data)
        entity_vocab = Vocab.from_dict(entity_data)
        logger.info(
            "Vocabs loaded: text_vocab_size=%d, entity_vocab_size=%d",
            len(text_vocab),
            len(entity_vocab),
        )
        return text_vocab, entity_vocab

    if not build_if_missing:
        raise FileNotFoundError(
            f"Vocab files not found under {vocabs_dir}, and build_if_missing=False."
        )

    # ---------- 情况 2：从训练数据构建新词表 ----------
    samples = list(_iter_preprocessed_samples(train_loader))
    logger.info("Building vocabs from %d preprocessed samples.", len(samples))

    text_vocab_cfg = getattr(config, "text_vocab", None)
    entity_vocab_cfg = getattr(config, "entity_vocab", None)

    text_vocab = build_text_vocab(samples, cfg=text_vocab_cfg)
    entity_vocab = build_entity_vocab(samples, cfg=entity_vocab_cfg)

    with text_path.open("w", encoding="utf-8") as f:
        json.dump(text_vocab.to_dict(), f, ensure_ascii=False, indent=2)
    with entity_path.open("w", encoding="utf-8") as f:
        json.dump(entity_vocab.to_dict(), f, ensure_ascii=False, indent=2)

    logger.info(
        "Vocabs built and saved: text_vocab_size=%d, entity_vocab_size=%d",
        len(text_vocab),
        len(entity_vocab),
    )
    return text_vocab, entity_vocab


# ============================================================
# Phase 3.5：PreprocessedSample → BatchEncoding
# ============================================================


def build_batcher(
    config: Any,
    text_vocab: Vocab,
    entity_vocab: Vocab,
) -> Batcher:
    """
    @brief 从实验配置与词表构建 Batcher 实例。
           Build a Batcher from experiment config and vocabularies.
    @param config 实验配置对象，需包含 batching 字段。
           Experiment config expected to have a 'batching' field.
    @param text_vocab 文本词表。Text vocabulary.
    @param entity_vocab 实体词表。Entity vocabulary.
    @return Batcher 实例。Batcher instance.
    """
    batching_cfg = getattr(config, "batching", None)
    if batching_cfg is None:
        batching_cfg = BatchingConfig()
    if not isinstance(batching_cfg, BatchingConfig):
        raise TypeError(
            f"config.batching must be BatchingConfig, got {type(batching_cfg)!r}"
        )

    batcher = Batcher(
        text_vocab=text_vocab,
        entity_vocab=entity_vocab,
        cfg=batching_cfg,
    )
    logger.info(
        "Batcher built: max_text_len=%d, max_entities=%d, max_context_len=%d",
        batching_cfg.max_text_len,
        batching_cfg.max_entities,
        batching_cfg.max_text_len,
    )
    return batcher


def build_batched_dataloaders(
    config: ExperimentConfig,
    text_vocab: Vocab,
    entity_vocab: Vocab,
    train_preprocessed_loader: DataLoader[Sequence[PreprocessedSample]],
) -> DataLoader[BatchEncoding]:
    """
    @brief 基于预处理 DataLoader 和词表，构建输出 BatchEncoding 的 DataLoader。
           Build a DataLoader yielding BatchEncoding from preprocessed samples.
    @param config ExperimentConfig 实验配置。Experiment configuration.
    @param text_vocab 文本词表。Text vocabulary.
    @param entity_vocab 实体词表。Entity vocabulary.
    @param train_preprocessed_loader 预处理级别的 DataLoader。
           DataLoader at PreprocessedSample level.
    @return 一个 DataLoader，其每个元素为 BatchEncoding。
            A DataLoader whose elements are BatchEncoding.
    """
    ds_cfg = config.dataset
    batcher = build_batcher(config, text_vocab, entity_vocab)

    dataset = getattr(train_preprocessed_loader, "dataset", None)
    if dataset is None:
        raise ValueError(
            "train_preprocessed_loader must have a .dataset attribute "
            "to rebuild batched DataLoader."
        )

    batched_loader: DataLoader[BatchEncoding] = DataLoader(
        dataset,
        batch_size=ds_cfg.batch_size,
        shuffle=ds_cfg.shuffle,
        collate_fn=batcher,
    )
    return batched_loader


# ============================================================
# Phase 4：模型构建（新 KAN + 显式嵌入）
# ============================================================


def build_kan_with_embeddings(
    config: ExperimentConfig,
    text_vocab: Vocab,
    entity_vocab: Vocab,
) -> KAN:
    """
    @brief 从 ExperimentConfig 与词表构建带嵌入的一站式 KAN 模型。
           Build a KAN model with integrated embeddings from ExperimentConfig
           and vocabularies.
    """
    # -------- 1) 先构造 KANConfig --------
    TextEncCfg = kan.models.kan.TextEncoderConfig
    TransformerEncCfg = kan.models.transformer_encoder.TransformerEncoderConfig
    KnowledgeEncCfg = kan.models.knowledge_encoder.KnowledgeEncoderConfig

    raw_text_cfg = config.text_encoder
    if isinstance(raw_text_cfg, TextEncCfg):
        text_cfg = raw_text_cfg
    elif isinstance(raw_text_cfg, TransformerEncCfg):
        text_cfg = TextEncCfg(encoder=raw_text_cfg)
    else:
        raise TypeError(
            f"ExperimentConfig.text_encoder 类型非法: {type(raw_text_cfg)!r}，"
            "期望 TransformerEncoderConfig 或 TextEncoderConfig。"
        )

    raw_kg_cfg = config.knowledge_encoder
    if isinstance(raw_kg_cfg, KnowledgeEncCfg):
        kg_cfg = raw_kg_cfg
    elif isinstance(raw_kg_cfg, TransformerEncCfg):
        kg_cfg = KnowledgeEncCfg(encoder=raw_kg_cfg)
    else:
        raise TypeError(
            f"ExperimentConfig.knowledge_encoder 类型非法: {type(raw_kg_cfg)!r}，"
            "期望 KnowledgeEncoderConfig 或 TransformerEncoderConfig。"
        )

    kan_cfg = KANConfig(
        text=text_cfg,
        knowledge=kg_cfg,
        attention=config.attention,
        num_classes=getattr(getattr(config, "model", None), "num_classes", 2),
        final_dropout=getattr(getattr(config, "model", None), "final_dropout", 0.1),
        use_entity_contexts=getattr(
            getattr(config, "model", None), "use_entity_contexts", True
        ),
    )

    # -------- 2) 文本嵌入 TextEmbeddingConfig --------
    te_cfg = getattr(config, "text_embedding", None)
    if te_cfg is None:
        te_cfg = TextEmbeddingConfig(
            vocab_size=len(text_vocab),
            d_model=kan_cfg.text.encoder.d_model,
            max_len=config.preprocess.max_tokens,
            dropout=0.1,
        )
    else:
        if getattr(te_cfg, "vocab_size", 0) <= 0:
            te_cfg.vocab_size = len(text_vocab)  # type: ignore[attr-defined]
        if getattr(te_cfg, "d_model", 0) <= 0:
            te_cfg.d_model = kan_cfg.text.encoder.d_model  # type: ignore[attr-defined]
        if getattr(te_cfg, "max_len", 0) <= 0:
            te_cfg.max_len = config.preprocess.max_tokens  # type: ignore[attr-defined]

    # -------- 3) 实体嵌入 EntityEmbeddingConfig（这里是关键修复点） --------
    ee_cfg = getattr(config, "entity_embedding", None)
    if ee_cfg is None:
        ee_cfg = EntityEmbeddingConfig(
            vocab_size=len(entity_vocab),
            d_model=kan_cfg.attention.d_model,
            dropout=0.1,
            share_entity_context_embeddings=True,
            context_pooling="mean",
        )
    else:
        if getattr(ee_cfg, "vocab_size", 0) <= 0:
            ee_cfg.vocab_size = len(entity_vocab)  # type: ignore[attr-defined]
        if getattr(ee_cfg, "d_model", 0) <= 0:
            ee_cfg.d_model = kan_cfg.attention.d_model  # type: ignore[attr-defined]

    # -------- 4) 从 vocab 构造嵌入层 --------
    text_embedding = TextEmbedding.from_vocab(
        text_vocab, te_cfg.d_model, te_cfg.max_len, te_cfg.dropout
    )
    entity_embedding = EntityEmbedding.from_vocab(
        entity_vocab,
        ee_cfg.d_model,
        ee_cfg.dropout,
        ee_cfg.share_entity_context_embeddings,
        ee_cfg.context_pooling,
    )

    model = KAN(
        config=kan_cfg,
        text_embedding=text_embedding,
        entity_embedding=entity_embedding,
    )
    logger.info("KAN built from ExperimentConfig and vocabs.")
    return model


# ============================================================
# Phase 5：优化器 / 学习率调度器（可选）
# ============================================================


def build_optimizer(config: Any, model: nn.Module) -> Optimizer:
    """
    @brief 基于 ExperimentConfig.training 构建 AdamW 优化器。
           Build an AdamW optimizer from ExperimentConfig.training.
    @param config 实验配置对象，需包含 training 字段。
           Experiment config expected to have a 'training' field.
    @param model 待训练模型。Model to optimize.
    @return 构造好的 Optimizer 实例。Constructed Optimizer instance.
    """
    training_cfg = getattr(config, "training", None)
    if training_cfg is None:
        raise ValueError("config.training is required to build optimizer.")

    lr = getattr(training_cfg, "learning_rate", 1e-3)
    weight_decay = getattr(training_cfg, "weight_decay", 0.0)

    logger.info(
        "Building AdamW optimizer: lr=%.3e, weight_decay=%.2e",
        lr,
        weight_decay,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    return optimizer


def build_scheduler(config: Any, optimizer: Optimizer) -> Optional[_LRScheduler]:
    """
    @brief 基于 ExperimentConfig.training 构建学习率调度器（可选）。
           Build an optional LR scheduler from ExperimentConfig.training.
    @param config 实验配置对象，需包含 training 字段。
           Experiment config expected to have a 'training' field.
    @param optimizer 已构建好的优化器。Optimizer instance.
    @return 若启用调度器则返回 _LRScheduler，否则返回 None。
            _LRScheduler if enabled, else None.
    @note 当前实现提供线性 warmup 调度策略：
          - step < warmup_steps: 线性从 0 → 1
          - step >= warmup_steps: 学习率保持基准值
          This implementation uses a linear warmup schedule:
          - step < warmup_steps: linear 0 → 1
          - step >= warmup_steps: LR stays at base value.
    """
    training_cfg = getattr(config, "training", None)
    if training_cfg is None:
        logger.info("No config.training provided; not using LR scheduler.")
        return None

    warmup_steps = getattr(training_cfg, "warmup_steps", 0)
    if warmup_steps <= 0:
        logger.info("warmup_steps <= 0; no LR scheduler will be used.")
        return None

    logger.info(
        "Using linear warmup scheduler: warmup_steps=%d",
        warmup_steps,
    )

    def lr_lambda(step: int) -> float:
        if step <= 0:
            return 0.0
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
