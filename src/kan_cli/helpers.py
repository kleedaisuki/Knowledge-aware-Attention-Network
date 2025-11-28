from kan import get_logger

logger = get_logger(__name__)


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
           Dataset wrapper for preprocessed samples, compatible with PyTorch DataLoader.
    """

    def __init__(self, samples: list[PreprocessedSample]) -> None:
        """
        @brief 保存预处理后的样本列表。
               Store a list of preprocessed samples.
        @param samples 预处理后的样本列表。List of preprocessed samples.
        """
        self._samples = samples

    def __len__(self) -> int:
        """
        @brief 返回样本数量。Return number of samples.
        @return 数据集中样本数。Number of samples in the dataset.
        """
        return len(self._samples)

    def __getitem__(self, idx: int) -> PreprocessedSample:
        """
        @brief 按索引获取单条样本。Get one sample by index.
        @param idx 样本索引。Sample index.
        @return 对应的 PreprocessedSample。Corresponding PreprocessedSample.
        """
        return self._samples[idx]


def _build_preprocessed_dataset(
    cfg_dataset: DatasetConfig,
    cfg_preprocess: PreprocessConfig,
    cfg_kg: KnowledgeGraphConfig,
) -> PreprocessedDataset:
    """
    @brief 从配置构建完整的预处理数据集：CSV → NewsSample → PreprocessedSample。
           Build a fully preprocessed dataset from configs: CSV → NewsSample → PreprocessedSample.
    @param cfg_dataset 数据集配置。Dataset configuration.
    @param cfg_preprocess 预处理配置。Preprocess configuration.
    @param cfg_kg 知识图谱配置。Knowledge graph configuration.
    @return PreprocessedDataset 实例。PreprocessedDataset instance.
    """
    # 1) 读取原始 CSV 样本
    dataset = NewsDataset(cfg_dataset)
    logger.info(
        "Building preprocessed dataset from %s (num_samples=%d)",
        cfg_dataset.csv_path,
        len(dataset.samples),
    )

    # 2) 构造 KG 客户端（如启用）
    kg_client = None
    if cfg_preprocess.enable_kg:
        kg_client = KnowledgeGraphClient(cfg_kg)

    preprocessor = Preprocessor(cfg_preprocess, kg_client=kg_client)

    # 3) 对全部样本做一次性预处理（简单实现，后续可改为惰性）
    def _get_samples_from_news_dataset(dataset: NewsDataset) -> Iterable[NewsSample]:
        for sample in dataset.samples:
            yield sample

    preprocessed: list[PreprocessedSample] = preprocessor.preprocess_batch(
        _get_samples_from_news_dataset(dataset)
    )

    logger.info("Preprocessed dataset built: %d samples", len(preprocessed))
    return PreprocessedDataset(preprocessed)


def build_all_dataloaders(
    config: Any,
) -> tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    @brief 根据 ExperimentConfig 构建 (train, val, test) 三个 DataLoader。
           Build (train, val, test) DataLoaders from an ExperimentConfig-like object.
    @param config 实验配置，需至少包含 dataset / preprocess / kg 字段。
           Experiment config, expected to have dataset / preprocess / kg attributes.
    @return (train_loader, val_loader, test_loader) 三元组，当前仅构建 train。
            Triple (train_loader, val_loader, test_loader); currently only train is built.
    @note 当前实现假定单一 CSV 作为训练集，不区分验证 / 测试集。
          This implementation assumes a single CSV as training data, without separate
          validation/test splits yet.
    """
    ds_cfg = config.dataset
    pp_cfg = config.preprocess
    kg_cfg = config.kg

    preprocessed_dataset = _build_preprocessed_dataset(ds_cfg, pp_cfg, kg_cfg)

    train_loader = DataLoader(
        preprocessed_dataset,
        batch_size=ds_cfg.batch_size,
        shuffle=ds_cfg.shuffle,
        collate_fn=identity_collate,
    )

    # 未来扩展：可以根据 config.dataset_{train,val,test} 构建更多 DataLoader
    val_loader: Optional[DataLoader] = None
    test_loader: Optional[DataLoader] = None

    return train_loader, val_loader, test_loader


def _iter_preprocessed_samples(
    train_loader: Iterable[Any],
) -> Iterable[PreprocessedSample]:
    """
    @brief 辅助函数：从任意可迭代 train_loader 中抽取 PreprocessedSample。
           Helper to extract PreprocessedSample objects from a generic iterable loader.
    @param train_loader 训练数据迭代器，元素可以是 PreprocessedSample 或其列表批次。
           Training data iterable; elements can be PreprocessedSample or batches of them.
    @return 逐个产出 PreprocessedSample 的生成器。Generator yielding PreprocessedSample.
    @throws TypeError 如元素类型不符合预期。
            Raises TypeError if element types are unsupported.
    """
    for item in train_loader:
        if isinstance(item, PreprocessedSample):
            # 单条样本
            yield item
        elif isinstance(item, (list, tuple)):
            # batch 内的每一个元素
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
                "build_or_load_vocabs currently expects train_loader to yield "
                "PreprocessedSample instances or batches thereof; got "
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
    @param train_loader 训练数据迭代器，通常是 DataLoader，也可直接是预处理数据集。
           Training data iterable, usually a DataLoader, or a preprocessed dataset.
    @param build_if_missing 若为 True，在无现有词表时从训练数据构建新词表。
           If True, build new vocabs from training data when files are missing.
    @return (text_vocab, entity_vocab) 二元组。
            A pair (text_vocab, entity_vocab).
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
        return text_vocab, entity_vocab

    if not build_if_missing:
        raise FileNotFoundError(
            f"Vocab files not found under {vocabs_dir}, and build_if_missing=False."
        )

    # ---------- 情况 2：从训练数据构建新词表 ----------
    # 优先从 DataLoader.dataset 取出预处理后的数据集本体，避免 default_collate
    if hasattr(train_loader, "dataset"):
        dataset = train_loader.dataset
        logger.info(
            "Building new vocabs from dataset attached to train_loader: %s",
            type(dataset).__name__,
        )
    else:
        # 兜底：如果传进来的本身就是一个 iterable of PreprocessedSample
        dataset = train_loader
        logger.info(
            "Building new vocabs from generic training iterable: %s",
            type(dataset).__name__,
        )

    # 这里直接遍历“样本级别”的 dataset，不再触发 DataLoader 的 default_collate
    samples = list(dataset)
    if not samples:
        raise RuntimeError("No training samples found when building vocabularies.")

    text_vocab_cfg = getattr(config, "text_vocab", None)
    entity_vocab_cfg = getattr(config, "entity_vocab", None)

    text_vocab = build_text_vocab(samples, cfg=text_vocab_cfg)
    entity_vocab = build_entity_vocab(samples, cfg=entity_vocab_cfg)

    # ---------- 持久化到 JSON ----------
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
    @example
        >>> batcher = build_batcher(exp_cfg, text_vocab, entity_vocab)
        >>> batch = batcher.collate_fn(list_of_preprocessed_samples)
    """
    batching_cfg = getattr(config, "batching", None)
    if batching_cfg is None:
        batching_cfg = BatchingConfig()
    elif not isinstance(batching_cfg, BatchingConfig):
        # 尽量容忍外部传入的“相似对象”，通过字段拷贝构造真正的 BatchingConfig
        batching_cfg = BatchingConfig(
            **{
                field: getattr(batching_cfg, field)
                for field in BatchingConfig.__annotations__.keys()
                if hasattr(batching_cfg, field)
            }
        )

    return Batcher(
        text_vocab=text_vocab,
        entity_vocab=entity_vocab,
        cfg=batching_cfg,
    )


def build_model_from_config(config: ExperimentConfig) -> KAN:
    """
    @brief 从 ExperimentConfig 构建 KAN 模型。
           Build a KAN model instance from an ExperimentConfig-like object.
    @param config 实验配置对象，需至少包含 text_encoder / knowledge_encoder / attention。
           Experiment config expected to have text_encoder / knowledge_encoder / attention.
    @return 构造好的 KAN 模型实例。Constructed KAN model instance.
    @note 本函数只负责 KAN 主干结构，不包含 ID → embedding 的部分；
          推荐在上层组合 TextEmbedding / EntityEmbedding 与 KAN。
          This function builds only the core KAN network, without ID→embedding;
          higher-level code should compose TextEmbedding / EntityEmbedding with KAN.
    """
    TextEncoderConfig = kan.models.kan.TextEncoderConfig
    TransformerEncoderConfig = kan.models.transformer_encoder.TransformerEncoderConfig
    KnowledgeEncoderConfig = kan.models.knowledge_encoder.KnowledgeEncoderConfig

    # ---- 1) 处理 text encoder：统一成 TextEncoderConfig ----

    raw_text_cfg = config.text_encoder

    if isinstance(raw_text_cfg, TextEncoderConfig):
        text_cfg = raw_text_cfg
    elif isinstance(raw_text_cfg, TransformerEncoderConfig):
        text_cfg = TextEncoderConfig(encoder=raw_text_cfg)
    else:
        raise TypeError(
            f"ExperimentConfig.text_encoder 类型非法: {type(raw_text_cfg)}，"
            "期望 TransformerEncoderConfig 或 TextEncoderConfig。"
        )

    # ---- 2) 处理 knowledge encoder：统一成 KnowledgeEncoderConfig ----
    raw_kg_cfg = config.knowledge_encoder

    if isinstance(raw_kg_cfg, KnowledgeEncoderConfig):
        kg_cfg = raw_kg_cfg
    elif isinstance(raw_kg_cfg, TransformerEncoderConfig):
        # 如果你将来允许直接用一个 TransformerEncoderConfig 作为知识编码器，也可以这样包装
        kg_cfg = KnowledgeEncoderConfig(encoder=raw_kg_cfg)
    else:
        raise TypeError(
            f"ExperimentConfig.knowledge_encoder 类型非法: {type(raw_kg_cfg)}，"
            "期望 KnowledgeEncoderConfig 或 TransformerEncoderConfig。"
        )

    # ---- 3) 组装 KANConfig ----
    kan_cfg = KANConfig(
        text=text_cfg,
        knowledge=kg_cfg,
        attention=config.attention,
        # num_classes / final_dropout 使用 KANConfig 自身默认值，
        # 如需覆盖，可在 ExperimentConfig 中扩展相应字段后在此读取。
    )
    return KAN(kan_cfg)


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
        logger.info("No training config; scheduler disabled.")
        return None

    use_scheduler = getattr(training_cfg, "use_scheduler", False)
    warmup_steps = getattr(training_cfg, "warmup_steps", 0)

    if not use_scheduler or warmup_steps <= 0:
        logger.info(
            "Scheduler disabled (use_scheduler=%s, warmup_steps=%d).",
            use_scheduler,
            warmup_steps,
        )
        return None

    def lr_lambda(step: int) -> float:
        if step <= 0:
            return 0.0
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0

    logger.info("Using linear warmup scheduler: warmup_steps=%d", warmup_steps)
    return LambdaLR(optimizer, lr_lambda=lr_lambda)
