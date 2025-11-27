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
from kan_cli.batching import StringHashEmbedding

logger = kan.get_logger(__name__)


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

    logger.info(f"使用模型: {model_path}")
    logger.info(f"对数据集进行预测: {data_path}")

    # 1. 加载配置（如果提供）
    exp_cfg = kan.load_experiment_config(cfg_path)  # type: ignore[attr-defined]
    if not isinstance(exp_cfg, kan.ExperimentConfig):  # type: ignore[attr-defined]
        raise RuntimeError(
            "KAN CLI: load_experiment_config must return ExperimentConfig."
        )

    # 2. 构建配置 dataclass
    dataset_cfg = exp_cfg.dataset
    dataset_cfg.csv_path = data_path  # 覆盖 path
    dataset_cfg.shuffle = False

    preprocess_cfg = exp_cfg.preprocess
    kg_cfg = exp_cfg.kg
    text_encoder_cfg = exp_cfg.text_encoder
    knowledge_encoder_cfg = exp_cfg.knowledge_encoder
    attention_cfg = exp_cfg.attention
    evaluation_cfg = (
        exp_cfg.evaluation
    )  # 若你准备在这里放 batch_size / threshold 等，可以后续使用

    # 3. 构建数据组件
    NewsDataset = kan.NewsDataset  # type: ignore[attr-defined]
    Preprocessor = kan.Preprocessor  # type: ignore[attr-defined]
    KnowledgeGraphClient = kan.KnowledgeGraphClient  # type: ignore[attr-defined]
    TextEncoderConfig = kan.models.kan.TextEncoderConfig  # type: ignore[attr-defined]
    KANConfig = kan.KANConfig

    dataset = NewsDataset(dataset_cfg)
    kg_client = KnowledgeGraphClient(kg_cfg) if preprocess_cfg.enable_kg else None
    preprocessor = Preprocessor(preprocess_cfg, kg_client=kg_client)

    text_cfg = TextEncoderConfig(encoder=text_encoder_cfg)
    model_cfg = KANConfig(
        text=text_cfg,
        knowledge=knowledge_encoder_cfg,
        attention=attention_cfg,
    )

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
    embed_dim = model_cfg.text.encoder.d_model
    embedder = StringHashEmbedding(embed_dim)

    # 4. 遍历数据，推理概率
    all_ids: List[int] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for batch in iter_batches_for_inference(
            dataset=dataset,
            preprocessor=preprocessor,
            embed_dim=embed_dim,
            embedder=embedder,
            with_labels=False,
        ):
            ids = batch.pop("ids")
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v
            pos_logits = wrapped_model(**batch)
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

    logger.info(f"预测完成，已写出结果: {output_path}")
