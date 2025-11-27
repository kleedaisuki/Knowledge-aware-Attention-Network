"""
@file config.py
@brief 实验配置加载模块：从 JSON 读取整体配置并构造各子模块的 dataclass。
       Experiment configuration loader: build all sub-config dataclasses from JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict

from kan.utils.logging import get_logger
from kan.data.datasets import DatasetConfig
from kan.data.preprocessing import PreprocessConfig
from kan.data.knowledge_graph import KnowledgeGraphConfig
from kan.models.transformer_encoder import TransformerEncoderConfig
from kan.models.knowledge_encoder import KnowledgeEncoderConfig
from kan.models.attention import KnowledgeAttentionConfig
from kan.training.trainer import TrainingConfig
from kan.training.evaluator import EvaluationConfig

logger = get_logger(__name__)


# ============================================================
# 顶层 ExperimentConfig 聚合 Aggregate config
# ============================================================


@dataclass
class ExperimentConfig:
    """
    @brief 一次完整实验/训练运行所需的全部配置聚合。
           Aggregate configuration for a full experiment / training run.
    @param dataset 数据集相关配置。Dataset configuration.
    @param preprocess 预处理配置。Preprocessing configuration.
    @param kg 知识图谱客户端配置。Knowledge graph client configuration.
    @param text_encoder 文本 Transformer 编码器配置。Text Transformer encoder config.
    @param knowledge_encoder 知识编码器配置。Knowledge encoder config.
    @param attention 知识注意力层配置。Knowledge attention config.
    @param training 训练流程配置。Training pipeline config.
    @param evaluation 评估 / 推理流程配置。Evaluation / inference pipeline config.
    """

    dataset: DatasetConfig
    preprocess: PreprocessConfig
    kg: KnowledgeGraphConfig
    text_encoder: TransformerEncoderConfig
    knowledge_encoder: KnowledgeEncoderConfig
    attention: KnowledgeAttentionConfig
    training: TrainingConfig
    evaluation: EvaluationConfig


# ============================================================
# 内部工具：字段过滤 Helper: filter kwargs
# ============================================================


def _filter_kwargs(cls: type, raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    @brief 过滤掉 JSON 中多余的字段，只保留 dataclass 已声明的字段。
           Filter out unknown keys from JSON, keeping only dataclass fields.
    @param cls 目标 dataclass 类型。Target dataclass type.
    @param raw JSON 解析出的原始字典。Raw dict from JSON.
    @return 可安全传入 cls(**kwargs) 的字典。Safe kwargs for cls(**kwargs).
    """
    if not raw:
        return {}
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in valid}


# ============================================================
# 解析 JSON → ExperimentConfig
# ============================================================


def _build_experiment_config(data: Dict[str, Any]) -> ExperimentConfig:
    """
    @brief 从原始 JSON 字典构造 ExperimentConfig。
           Build ExperimentConfig from raw JSON dictionary.
    @param data 顶层 JSON 字典。Top-level JSON dict.
    @return ExperimentConfig 实例。ExperimentConfig instance.
    """

    # ---------- dataset ----------
    dataset_cfg = DatasetConfig(
        **_filter_kwargs(DatasetConfig, data.get("dataset", {}))
    )

    # ---------- preprocess ----------
    preprocess_cfg = PreprocessConfig(
        **_filter_kwargs(PreprocessConfig, data.get("preprocess", {}))
    )

    # ---------- knowledge graph ----------
    kg_cfg = KnowledgeGraphConfig(
        **_filter_kwargs(KnowledgeGraphConfig, data.get("kg", {}))
    )

    # ---------- text encoder (news encoder) ----------
    text_encoder_cfg = TransformerEncoderConfig(
        **_filter_kwargs(TransformerEncoderConfig, data.get("text_encoder", {}))
    )

    # ---------- knowledge encoder (entities + contexts) ----------
    ke_raw = data.get("knowledge_encoder", {}) or {}

    # 嵌套字段 encoder 需要先专门构造 TransformerEncoderConfig
    # Nested field "encoder" should be built explicitly
    encoder_raw = ke_raw.get("encoder", {})
    if isinstance(encoder_raw, dict) and encoder_raw:
        encoder_cfg = TransformerEncoderConfig(
            **_filter_kwargs(TransformerEncoderConfig, encoder_raw)
        )
        ke_kwargs = _filter_kwargs(KnowledgeEncoderConfig, ke_raw)
        # 避免重复传入 encoder 字段
        ke_kwargs.pop("encoder", None)
        knowledge_encoder_cfg = KnowledgeEncoderConfig(encoder=encoder_cfg, **ke_kwargs)
    else:
        # 没有提供 encoder 子配置，则使用 KnowledgeEncoderConfig 自带默认 encoder
        knowledge_encoder_cfg = KnowledgeEncoderConfig(
            **_filter_kwargs(KnowledgeEncoderConfig, ke_raw)
        )

    # ---------- attention (N-E & N-E²C) ----------
    att_raw = data.get("attention")

    if isinstance(att_raw, dict) and att_raw:
        attention_cfg = KnowledgeAttentionConfig(
            **_filter_kwargs(KnowledgeAttentionConfig, att_raw)
        )
    else:
        # 若未在 JSON 中显式给出 attention，则使用与文本编码器一致的维度/头数
        # If not specified, align with text encoder's d_model/nhead
        attention_cfg = KnowledgeAttentionConfig(
            d_model=text_encoder_cfg.d_model,
            nhead=text_encoder_cfg.nhead,
        )

    # ---------- training ----------
    training_cfg = TrainingConfig(
        **_filter_kwargs(TrainingConfig, data.get("training", {}))
    )

    # ---------- evaluation ----------
    # 若 JSON 中没有 evaluation 字段，则使用 EvaluationConfig 的默认参数
    # If "evaluation" is not present in JSON, EvaluationConfig defaults are used.
    evaluation_cfg = EvaluationConfig(
        **_filter_kwargs(EvaluationConfig, data.get("evaluation", {}))
    )

    return ExperimentConfig(
        dataset=dataset_cfg,
        preprocess=preprocess_cfg,
        kg=kg_cfg,
        text_encoder=text_encoder_cfg,
        knowledge_encoder=knowledge_encoder_cfg,
        attention=attention_cfg,
        training=training_cfg,
        evaluation=evaluation_cfg,
    )


# ============================================================
# 路径解析：name 或 path → 真实 JSON 路径
# ============================================================


def _resolve_config_path(name_or_path: str) -> Path:
    """
    @brief 将配置名或路径解析为实际 JSON 文件路径。
           Resolve a config name or path into an actual JSON file path.
    @param name_or_path 可以是绝对/相对路径；也可以是不带后缀的配置名。
           May be an absolute/relative path, or a config name without suffix.
    @return 指向 .json 文件的 Path 对象。Path object pointing to a .json file.
    @note 若传入 'baseline'，则会在 'src/kan/configs/baseline.json' 下查找。
          If 'baseline' is given, it tries 'src/kan/configs/baseline.json'.
    """
    p = Path(name_or_path)

    # 1) 直接是一个存在的文件
    if p.is_file():
        return p.resolve()

    # 2) 作为名字，去 kan/configs 下寻找
    #    Locate under package's configs directory as a name.
    base_dir = Path(__file__).resolve().parents[2]  # .../src/
    configs_dir = base_dir / "kan" / "configs"

    if not name_or_path.endswith(".json"):
        filename = name_or_path + ".json"
    else:
        filename = name_or_path

    candidate = configs_dir / filename
    if candidate.is_file():
        return candidate.resolve()

    raise FileNotFoundError(
        f"Config JSON not found: {name_or_path} " f"(tried as path and as {candidate})"
    )


# ============================================================
# 对外主入口：加载配置 Public API
# ============================================================


def load_experiment_config(name_or_path: str) -> ExperimentConfig:
    """
    @brief 从给定 JSON 路径或配置名加载 ExperimentConfig。
           Load ExperimentConfig from JSON path or config name.
    @param name_or_path JSON 文件路径，或 configs 下的配置名（可不带 .json）。
           JSON file path, or config name under 'kan/configs' (without .json).
    @return ExperimentConfig 实例。ExperimentConfig instance.
    @example
        >>> cfg = load_experiment_config("baseline")
        >>> cfg = load_experiment_config("src/kan/configs/debug.json")
    """
    path = _resolve_config_path(name_or_path)
    logger.info("Loading experiment config from: %s", path)

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Top-level JSON must be an object/dict, got: {type(raw)!r}")

    cfg = _build_experiment_config(raw)
    logger.info("ExperimentConfig loaded successfully.")
    return cfg


# 为简洁起见，提供一个别名 load_config
# Provide a short alias load_config for convenience
load_config = load_experiment_config
