"""
@file predict.py
@brief `kan predict` 子命令实现：对不含标签的 CSV 数据进行推理，并输出 id,prob。
       Implementation of `kan predict` sub-command: run inference on unlabeled CSV
       data and output id,prob.
"""

from __future__ import annotations

import argparse
from typing import Any, List

import torch

import kan  # 顶层 API :contentReference[oaicite:7]{index=7}
from kan_cli.model_wrapper import KANForTrainer
from kan_cli.batching import iter_batches_for_inference


def cli_predict(args: argparse.Namespace) -> None:
    """
    @brief `kan predict` 子命令入口。
           Entry for `kan predict` sub-command.
    @param args 命令行参数，需包含 --model, --data，可选 --config, --output。
           CLI arguments, require --model, --data, optional --config, --output.
    """
    model_path: str = args.model
    data_path: str = args.data
    cfg_path: str | None = args.config
    output_path: str = args.output

    print(f"[KAN-CLI] 使用模型: {model_path}")
    print(f"[KAN-CLI] 对数据集进行预测: {data_path}")

    # 1. 加载配置（可选）
    if cfg_path is not None:
        if not hasattr(kan, "load_experiment_config"):
            raise RuntimeError("KAN CLI: 未找到 kan.load_experiment_config。")
        exp_cfg = kan.load_experiment_config(cfg_path)  # type: ignore[attr-defined]
    else:
        exp_cfg = {}

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
    dataset_cfg.csv_path = data_path  # type: ignore[attr-defined]
    dataset_cfg.shuffle = False

    preprocess_cfg = _sub("preprocess", PreprocessConfig)
    kg_cfg = _sub("knowledge_graph", KnowledgeGraphConfig)
    model_cfg = _sub("model", KANConfig)

    # 2. 构建数据组件
    NewsDataset = kan.NewsDataset  # type: ignore[attr-defined]
    Preprocessor = kan.Preprocessor  # type: ignore[attr-defined]
    KnowledgeGraphClient = kan.KnowledgeGraphClient  # type: ignore[attr-defined]

    dataset = NewsDataset(dataset_cfg)
    kg_client = KnowledgeGraphClient(kg_cfg) if preprocess_cfg.enable_kg else None
    preprocessor = Preprocessor(preprocess_cfg, kg_client=kg_client)

    # 3. 构建模型并加载权重
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

    # 4. 遍历数据，推理概率
    all_ids: List[int] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for batch in iter_batches_for_inference(
            dataset=dataset,
            preprocessor=preprocessor,
            embed_dim=embed_dim,
            with_labels=False,
        ):
            ids = batch.pop("ids")
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v

            logits = wrapped_model(**batch)  # type: ignore[arg-type]
            if logits.size(-1) == 1:
                pos_logits = logits.view(-1)
            else:
                pos_logits = logits[..., 1].view(-1)
            probs = torch.sigmoid(pos_logits)

            all_ids.extend(ids)
            all_probs.extend(probs.cpu().tolist())

    # 5. 输出 id,prob 到 CSV
    if hasattr(kan, "write_probability_csv"):
        kan.write_probability_csv(  # type: ignore[attr-defined]
            ids=all_ids,
            probs=all_probs,
            path=output_path,
        )
    else:
        import pandas as pd

        import pandas as pd

        df = pd.DataFrame({"id": all_ids, "prob": all_probs})
        df.to_csv(output_path, index=False)

    print(f"[KAN-CLI] 预测完成，已写出结果: {output_path}")
