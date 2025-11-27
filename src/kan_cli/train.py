"""
@file train.py
@brief `kan train` 子命令实现：加载配置、构建 KAN + 数据流，然后调用 Trainer 进行训练。
       Implementation of `kan train` sub-command: load config, build KAN + data
       pipeline, then run training via Trainer.
"""

from __future__ import annotations

import argparse
from importlib.resources import files
from typing import Any, Dict

import kan  # 顶层公共 API :contentReference[oaicite:5]{index=5}

from kan_cli.model_wrapper import KANForTrainer
from kan_cli.batching import iter_batches_for_training


def _resolve_default_config_path() -> str:
    """
    @brief 解析默认配置文件 kan/configs/default.json 的绝对路径。
           Resolve absolute path of default config kan/configs/default.json.
    """
    cfg_path = files("kan.configs") / "default.json"
    return str(cfg_path)


def _get_subconfig(exp_cfg: Any, name: str, cls: Any) -> Any:
    """
    @brief 从实验配置中提取子配置，可兼容 dict / dataclass / namespace。
           Extract sub-config from experiment config, compatible with dict/dataclass/namespace.
    @param exp_cfg 实验配置对象。Experiment config object.
    @param name 子配置名称，如 "dataset" / "preprocess"。
           Name of sub-config, e.g. "dataset" / "preprocess".
    @param cls 期望的 dataclass 类型。Expected dataclass type.
    @return 对应的 dataclass 实例。Corresponding dataclass instance.
    """
    # 优先属性访问
    if hasattr(exp_cfg, name):
        sub = getattr(exp_cfg, name)
    elif isinstance(exp_cfg, dict) and name in exp_cfg:
        sub = exp_cfg[name]
    else:
        sub = {}

    if isinstance(sub, cls):
        return sub
    if isinstance(sub, dict):
        return cls(**sub)  # type: ignore[call-arg]
    # 兜底：直接用默认构造
    return cls()  # type: ignore[call-arg]


def cli_train(args: argparse.Namespace) -> None:
    """
    @brief `kan train` 子命令入口。
           Entry for `kan train` sub-command.
    @param args 命令行参数。CLI arguments.
    """
    # 1. 确定配置路径
    if args.config is None:
        config_path = _resolve_default_config_path()
        print(f"[KAN-CLI] 使用默认配置文件: {config_path}")
    else:
        config_path = args.config
        print(f"[KAN-CLI] 使用自定义配置文件: {config_path}")

    # 2. 加载实验配置
    if not hasattr(kan, "load_experiment_config"):
        raise RuntimeError(
            "KAN CLI: 找不到 kan.load_experiment_config，请确认 utils.config 中实现并在 __init__ 导出。"
        )
    exp_cfg = kan.load_experiment_config(config_path)  # type: ignore[attr-defined]

    # 3. 构建各类子配置
    DatasetConfig = kan.DatasetConfig  # type: ignore[attr-defined]
    PreprocessConfig = kan.PreprocessConfig  # type: ignore[attr-defined]
    KnowledgeGraphConfig = kan.KnowledgeGraphConfig  # type: ignore[attr-defined]
    TrainingConfig = kan.TrainingConfig  # type: ignore[attr-defined]
    KANConfig = kan.KANConfig  # type: ignore[attr-defined]

    dataset_cfg = _get_subconfig(exp_cfg, "dataset", DatasetConfig)
    preprocess_cfg = _get_subconfig(exp_cfg, "preprocess", PreprocessConfig)
    kg_cfg = _get_subconfig(exp_cfg, "knowledge_graph", KnowledgeGraphConfig)
    training_cfg = _get_subconfig(exp_cfg, "training", TrainingConfig)
    model_cfg = _get_subconfig(exp_cfg, "model", KANConfig)

    # 4. 构建数据组件：NewsDataset + KnowledgeGraphClient + Preprocessor
    NewsDataset = kan.NewsDataset  # type: ignore[attr-defined]
    Preprocessor = kan.Preprocessor  # type: ignore[attr-defined]
    KnowledgeGraphClient = kan.KnowledgeGraphClient  # type: ignore[attr-defined]

    dataset = NewsDataset(dataset_cfg)
    kg_client = KnowledgeGraphClient(kg_cfg) if preprocess_cfg.enable_kg else None
    preprocessor = Preprocessor(preprocess_cfg, kg_client=kg_client)

    # 5. 构建 KAN 模型，并包装为 KANForTrainer
    KANModel = kan.KAN  # type: ignore[attr-defined]
    kan_model = KANModel(model_cfg)  # type: ignore[call-arg]

    # d_model 取自文本编码器配置
    embed_dim = model_cfg.text.encoder.d_model  # type: ignore[attr-defined]

    wrapped_model = KANForTrainer(kan_model)

    # 6. 构建训练数据迭代器
    train_data = iter_batches_for_training(
        dataset=dataset,
        preprocessor=preprocessor,
        embed_dim=embed_dim,
    )

    # 7. 构建 Trainer 并开始训练
    Trainer = kan.Trainer  # type: ignore[attr-defined]
    trainer = Trainer(model=wrapped_model, train_data=train_data, cfg=training_cfg)  # type: ignore[call-arg]

    print("[KAN-CLI] 开始训练...")
    trainer.train()  # type: ignore[call-arg]
    print("[KAN-CLI] 训练结束。")
