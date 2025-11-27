"""
@file evaluate.py
@brief `kan evaluate` 子命令实现：加载模型与验证集，计算二分类指标并可选输出概率 CSV。
       Implementation of `kan evaluate` sub-command: load model & validation set,
       compute binary classification metrics and optionally write probability CSV.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

import torch

import kan  # 顶层 API :contentReference[oaicite:6]{index=6}
from kan_cli.model_wrapper import KANForTrainer
from kan_cli.batching import iter_batches_for_inference


def cli_evaluate(args: argparse.Namespace) -> None:
    """
    @brief `kan evaluate` 子命令入口。
           Entry for `kan evaluate` sub-command.
    @param args 命令行参数，需包含 --model, --data，可选 --config, --metrics, --probs。
           CLI arguments, require --model, --data, optional --config, --metrics, --probs.
    """
    model_path: str = args.model
    data_path: str = args.data
    cfg_path: str | None = args.config
    metrics_path: str = args.metrics
    probs_path: str | None = args.probs

    print(f"[KAN-CLI] 评估模型: {model_path}")
    print(f"[KAN-CLI] 使用数据集: {data_path}")

    # 1. 加载配置（如果提供）
    if cfg_path is not None:
        if not hasattr(kan, "load_experiment_config"):
            raise RuntimeError("KAN CLI: 未找到 kan.load_experiment_config。")
        exp_cfg = kan.load_experiment_config(cfg_path)  # type: ignore[attr-defined]
    else:
        exp_cfg = {}

    # 2. 构建配置 dataclass
    DatasetConfig = kan.DatasetConfig  # type: ignore[attr-defined]
    PreprocessConfig = kan.PreprocessConfig  # type: ignore[attr-defined]
    KnowledgeGraphConfig = kan.KnowledgeGraphConfig  # type: ignore[attr-defined]
    KANConfig = kan.KANConfig  # type: ignore[attr-defined]

    def _sub(name: str, cls: Any) -> Any:
        if hasattr(exp_cfg, name):
            val = getattr(exp_cfg, name)
        elif isinstance(exp_cfg, dict) and name in exp_cfg:
            val = exp_cfg[name]
        else:
            val = {}
        if isinstance(val, cls):
            return val
        if isinstance(val, dict):
            return cls(**val)  # type: ignore[call-arg]
        return cls()  # type: ignore[call-arg]

    dataset_cfg = _sub("dataset", DatasetConfig)
    # 用命令行 data_path 覆盖 CSV 路径
    dataset_cfg.csv_path = data_path  # type: ignore[attr-defined]
    dataset_cfg.shuffle = False  # 评估不打乱

    preprocess_cfg = _sub("preprocess", PreprocessConfig)
    kg_cfg = _sub("knowledge_graph", KnowledgeGraphConfig)
    model_cfg = _sub("model", KANConfig)

    # 3. 构建数据组件
    NewsDataset = kan.NewsDataset  # type: ignore[attr-defined]
    Preprocessor = kan.Preprocessor  # type: ignore[attr-defined]
    KnowledgeGraphClient = kan.KnowledgeGraphClient  # type: ignore[attr-defined]

    dataset = NewsDataset(dataset_cfg)
    kg_client = KnowledgeGraphClient(kg_cfg) if preprocess_cfg.enable_kg else None
    preprocessor = Preprocessor(preprocess_cfg, kg_client=kg_client)

    # 4. 构建模型并加载 checkpoint
    KANModel = kan.KAN  # type: ignore[attr-defined]
    kan_model = KANModel(model_cfg)  # type: ignore[call-arg]
    wrapped_model = KANForTrainer(kan_model)

    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        wrapped_model.load_state_dict(state["model_state_dict"])
    else:
        wrapped_model.load_state_dict(state)

    wrapped_model.eval()

    embed_dim = model_cfg.text.encoder.d_model  # type: ignore[attr-defined]

    # 5. 遍历数据，收集 logits / labels / ids
    all_labels: List[int] = []
    all_probs: List[float] = []
    all_ids: List[int] = []

    with torch.no_grad():
        for batch in iter_batches_for_inference(
            dataset=dataset,
            preprocessor=preprocessor,
            embed_dim=embed_dim,
            with_labels=True,
        ):
            labels = batch.pop("labels").view(-1).to(torch.float32)
            ids = batch.pop("ids")
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v

            logits = wrapped_model(**batch)  # type: ignore[arg-type]
            # 二分类：我们取第二类（index=1）的 logit 做 prob
            if logits.size(-1) == 1:
                pos_logits = logits.view(-1)
            else:
                pos_logits = logits[..., 1].view(-1)

            probs = torch.sigmoid(pos_logits)

            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            all_ids.extend(ids)

    # 6. 计算指标
    if not hasattr(kan, "compute_binary_classification_metrics"):
        raise RuntimeError("KAN CLI: 未找到 compute_binary_classification_metrics。")
    metrics = kan.compute_binary_classification_metrics(  # type: ignore[attr-defined]
        y_true=all_labels,
        y_score=all_probs,
        threshold=0.5,
    )

    print("[KAN-CLI] 评估指标:")
    for k, v in metrics.__dict__.items():  # BinaryClassificationMetrics dataclass
        print(f"  {k}: {v}")

    # 7. 写 metrics 到 JSON
    metrics_dict = metrics.__dict__
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, ensure_ascii=False, indent=2)
    print(f"[KAN-CLI] 指标已写入: {metrics_path}")

    # 8. 可选写出概率 CSV
    if probs_path is not None:
        if hasattr(kan, "write_probability_csv"):
            kan.write_probability_csv(  # type: ignore[attr-defined]
                ids=all_ids,
                probs=all_probs,
                path=probs_path,
            )
        else:
            import pandas as pd

            df = pd.DataFrame({"id": all_ids, "prob": all_probs})
            df.to_csv(probs_path, index=False)
        print(f"[KAN-CLI] 概率结果已写入: {probs_path}")
