"""
@file train.py
@brief `kan train` 子命令实现：加载配置、构建 KAN + 数据流，然后调用 Trainer 进行训练。
       Implementation of `kan train` sub-command: load config, build KAN + data
       pipeline, then run training via Trainer.
"""

from __future__ import annotations

import argparse
from importlib.resources import files

import kan  # 顶层公共 API :contentReference[oaicite:5]{index=5}

logger = kan.get_logger(__name__)

from kan_cli.model_wrapper import KANForTrainer
from kan_cli.embedding import StringHashEmbedding
from kan_cli.batching import TrainingDataIterable


def _resolve_default_config_path() -> str:
    """
    @brief 解析默认配置文件 kan/configs/default.json 的绝对路径。
           Resolve absolute path of default config kan/configs/default.json.
    """
    cfg_path = files("kan.configs") / "default.json"
    return str(cfg_path)


def cli_train(args: argparse.Namespace) -> None:
    """
    @brief `kan train` 子命令入口。
           Entry for `kan train` sub-command.
    @param args 命令行参数。CLI arguments.
    """
    # 1. 确定配置路径
    if args.config is None:
        config_path = _resolve_default_config_path()
        logger.info(f"使用默认配置文件: {config_path}")
    else:
        config_path = args.config
        logger.info(f"使用自定义配置文件: {config_path}")

    # 2. 加载实验配置
    if not hasattr(kan, "load_experiment_config"):
        raise RuntimeError(
            "KAN CLI: 找不到 kan.load_experiment_config，请确认 utils.config 中实现并在 __init__ 导出。"
        )
    exp_cfg = kan.load_experiment_config(config_path)  # type: ignore[attr-defined]

    # 3. 构建各类子配置
    dataset_cfg = exp_cfg.dataset
    preprocess_cfg = exp_cfg.preprocess
    kg_cfg = exp_cfg.kg
    training_cfg = exp_cfg.training
    text_encoder_cfg = exp_cfg.text_encoder
    knowledge_encoder_cfg = exp_cfg.knowledge_encoder
    attention_cfg = exp_cfg.attention

    # 4. 构建数据组件：NewsDataset + KnowledgeGraphClient + Preprocessor
    NewsDataset = kan.NewsDataset  # type: ignore[attr-defined]
    Preprocessor = kan.Preprocessor  # type: ignore[attr-defined]
    KnowledgeGraphClient = kan.KnowledgeGraphClient  # type: ignore[attr-defined]

    dataset = NewsDataset(dataset_cfg)
    kg_client = KnowledgeGraphClient(kg_cfg) if preprocess_cfg.enable_kg else None
    preprocessor = Preprocessor(preprocess_cfg, kg_client=kg_client)

    # 5. 构建 KAN 模型，并包装为 KANForTrainer
    TextEncoderConfig = kan.models.kan.TextEncoderConfig  # type: ignore[attr-defined]
    KANConfig = kan.KANConfig  # type: ignore[attr-defined]

    text_cfg = TextEncoderConfig(encoder=text_encoder_cfg)
    model_cfg = KANConfig(
        text=text_cfg,
        knowledge=knowledge_encoder_cfg,
        attention=attention_cfg,
        # 其它字段使用默认值，或以后在 ExperimentConfig 里补充
    )

    # d_model 取自文本编码器配置
    NewsDataset = kan.NewsDataset  # type: ignore[attr-defined]
    Preprocessor = kan.Preprocessor  # type: ignore[attr-defined]
    KnowledgeGraphClient = kan.KnowledgeGraphClient  # type: ignore[attr-defined]

    dataset = NewsDataset(dataset_cfg)
    kg_client = KnowledgeGraphClient(kg_cfg) if preprocess_cfg.enable_kg else None
    preprocessor = Preprocessor(preprocess_cfg, kg_client=kg_client)

    KANModel = kan.KAN  # type: ignore[attr-defined]
    kan_model = KANModel(model_cfg)  # type: ignore[call-arg]

    embed_dim = model_cfg.text.encoder.d_model  # type: ignore[attr-defined]

    # 6. 构建训练数据迭代器
    NewsDataset = kan.NewsDataset  # type: ignore[attr-defined]
    Preprocessor = kan.Preprocessor  # type: ignore[attr-defined]
    KnowledgeGraphClient = kan.KnowledgeGraphClient  # type: ignore[attr-defined]

    dataset = NewsDataset(dataset_cfg)
    kg_client = KnowledgeGraphClient(kg_cfg) if preprocess_cfg.enable_kg else None
    preprocessor = Preprocessor(preprocess_cfg, kg_client=kg_client)

    embedder = StringHashEmbedding(embed_dim)
    train_data = TrainingDataIterable(
        dataset=dataset,
        preprocessor=preprocessor,
        embed_dim=embed_dim,
        embedder=embedder,
    )

    # 7. 构建 Trainer 并开始训练
    wrapped_model = KANForTrainer(kan_model)

    Trainer = kan.Trainer  # type: ignore[attr-defined]
    trainer = Trainer(model=wrapped_model, train_data=train_data, cfg=training_cfg)  # type: ignore[call-arg]

    logger.info("开始训练...")
    trainer.train()  # type: ignore[call-arg]
    logger.info("训练结束。")
