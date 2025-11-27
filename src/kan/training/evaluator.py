"""
@file evaluator.py
@brief 评估 / 推理流水线实现：加载模型、跑前向、计算指标并导出 CSV。
       Evaluation / inference pipeline: load model, run forward passes,
       compute metrics, and export CSV.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Mapping, Any, Optional, Union

import torch
from torch import nn, Tensor

from kan.utils.logging import get_logger
from kan.utils import metrics as metrics_utils  # type: ignore[import-not-found]

logger = get_logger(__name__)


@dataclass
class EvaluationConfig:
    """
    @brief 评估相关配置项。Configuration for evaluation / inference.
    @param device 设备字符串，例如 "cuda" 或 "cpu"。Device string, e.g., "cuda" or "cpu".
    @param seed 全局随机种子。Global random seed.
    @param checkpoint_path 要加载的模型权重文件路径（可选）。Path to checkpoint to load (optional).
    @param results_csv 预测结果 CSV 输出路径（id, prob）。CSV path for prediction results (id, prob).
    @param threshold 计算离散指标时的概率阈值。Threshold to binarize probabilities for metrics.
    @param has_labels 是否期望数据中包含 labels 字段。Whether batches are expected to include labels.
    @param compute_metrics 若为 True 且存在标签，则计算并返回指标。If True and labels present, compute metrics.
    """

    device: str = "cuda"
    seed: int = 42

    checkpoint_path: Optional[str] = None
    results_csv: str = "results.csv"

    threshold: float = 0.5
    has_labels: bool = True
    compute_metrics: bool = True


class Evaluator:
    """
    @brief 评估 / 推理用 Evaluator：负责加载 checkpoint、跑前向并调用 metrics 工具。
           Evaluator for evaluation / inference: loads checkpoint, runs forward passes,
           and delegates metrics / CSV writing to metrics utilities.
    """

    def __init__(
        self,
        model: nn.Module,
        data: Iterable[Mapping[str, Any]],
        cfg: EvaluationConfig,
    ) -> None:
        """
        @brief 构造 Evaluator 实例。
               Construct an Evaluator instance.
        @param model 已构建的 PyTorch 模型（例如 KAN）。Model instance (e.g., KAN).
        @param data 评估集 batch 序列，每个 batch 为一个 mapping。
               Iterable of evaluation batches; each batch is a mapping.
               约定键：
                 - "labels" (可选)：标签张量。
                 - "id" 或 "ids" (可选)：样本 id，用于 CSV 输出。
                 - 其它键将传入 model(**inputs)。
               Conventions:
                 - "labels" (optional): label tensor.
                 - "id" or "ids" (optional): sample ids for CSV output.
                 - Others forwarded to model(**inputs).
        @param cfg EvaluationConfig 评估配置。Evaluation configuration.
        """
        self.model = model
        self.data = data
        self.cfg = cfg

        self.device = self._prepare_device(cfg.device)
        self.model.to(self.device)

        self._set_seed(cfg.seed)

        # 可以选择性加载 checkpoint
        self._maybe_load_checkpoint(cfg.checkpoint_path)

    # --------------------------------------------------------
    # 辅助：设备 & 随机种子 & checkpoint
    # --------------------------------------------------------
    def _prepare_device(self, device_str: str) -> torch.device:
        """
        @brief 解析设备字符串并做 fallback（例如 cuda 不可用时退回 cpu）。
               Parse device string and fallback (e.g., to CPU if CUDA unavailable).
        @param device_str 配置中的 device 字符串。Device string from config.
        @return torch.device 对象。torch.device instance.
        """
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(
                "CUDA requested for evaluation but not available, falling back to CPU. "
                "Requested device=%s",
                device_str,
            )
            return torch.device("cpu")
        return torch.device(device_str)

    def _set_seed(self, seed: int) -> None:
        """
        @brief 设置全局随机种子以保证评估可复现。
               Set global random seed to make evaluation reproducible.
        @param seed 随机种子。Random seed value.
        """
        from kan.utils import seed as seed_utils  # type: ignore[import-not-found]

        seed_utils.set_global_seed(seed)  # type: ignore[attr-defined]
        logger.info("Global seed set via kan.utils.seed (evaluator): %d", seed)

    def _maybe_load_checkpoint(self, ckpt_path: Optional[str]) -> None:
        """
        @brief 如指定 checkpoint_path，则尝试加载模型权重。
               Load checkpoint if checkpoint_path is provided.
        @param ckpt_path checkpoint 路径，可为 None。Checkpoint path or None.
        @note 兼容 trainer.py 保存的格式：{'model_state_dict', 'training_config', 'epoch'}。
              Compatible with trainer.py's save format.
        """
        if not ckpt_path:
            logger.info("No checkpoint_path provided, using current model parameters.")
            return

        if not os.path.isfile(ckpt_path):
            logger.warning(
                "Checkpoint file not found, skip loading: %s",
                ckpt_path,
            )
            return

        try:
            state = torch.load(ckpt_path, map_location=self.device)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to load checkpoint from %s: %s", ckpt_path, e)
            return

        if isinstance(state, dict) and "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
            epoch = state.get("epoch", None)
            logger.info(
                "Checkpoint loaded from %s (epoch=%s)",
                ckpt_path,
                "unknown" if epoch is None else epoch,
            )
        else:
            # 兼容直接保存 state_dict 的情况
            try:
                self.model.load_state_dict(state)
                logger.info("Checkpoint loaded as plain state_dict from %s", ckpt_path)
            except Exception as e:  # noqa: BLE001
                logger.error(
                    "Loaded checkpoint object is not compatible with model: %s",
                    e,
                )

    # --------------------------------------------------------
    # 主评估逻辑
    # --------------------------------------------------------
    def evaluate(
        self,
    ) -> Optional[metrics_utils.BinaryClassificationMetrics]:
        """
        @brief 执行完整评估流程：跑前向、收集概率，必要时计算指标并写 CSV。
               Run full evaluation: forward passes, collect probabilities,
               optionally compute metrics, and write CSV.
        @return 若 cfg.compute_metrics 且存在标签，则返回 BinaryClassificationMetrics；
                否则返回 None。
                If cfg.compute_metrics and labels present, returns metrics; else None.
        """
        self.model.eval()

        all_ids: list[Any] = []
        all_probs: list[float] = []
        all_labels: list[int] = []

        has_any_labels = False

        with torch.no_grad():
            for batch in self.data:
                batch_ids = self._extract_ids(batch)
                batch_labels = self._extract_labels(batch)
                if batch_labels is not None:
                    has_any_labels = True

                logits = self._forward_batch(batch)

                # logits -> probs
                probs = torch.sigmoid(logits).view(-1)

                # 若 batch_ids 为空，用 0..n-1 的占位 id（不推荐实际比赛中这么做）
                if batch_ids is None:
                    batch_ids = list(range(len(probs)))

                if isinstance(batch_ids, torch.Tensor):
                    batch_ids_list = batch_ids.detach().cpu().tolist()
                else:
                    # 支持 list/tuple
                    batch_ids_list = list(batch_ids)

                if len(batch_ids_list) != probs.numel():
                    raise RuntimeError(
                        f"Number of ids ({len(batch_ids_list)}) "
                        f"does not match number of probs ({probs.numel()})."
                    )

                all_ids.extend(batch_ids_list)
                all_probs.extend(probs.detach().cpu().tolist())

                if batch_labels is not None:
                    if isinstance(batch_labels, torch.Tensor):
                        all_labels.extend(batch_labels.detach().cpu().view(-1).tolist())
                    else:
                        all_labels.extend(list(batch_labels))

        if not all_ids:
            logger.warning("No data was processed during evaluation.")
            return None

        # 写 CSV（无论是否有标签，都可以导出 id, prob）
        metrics_utils.write_probability_csv(all_ids, all_probs, self.cfg.results_csv)

        # 如有标签且需要指标，则计算并返回
        if (
            self.cfg.compute_metrics
            and self.cfg.has_labels
            and has_any_labels
            and len(all_labels) == len(all_probs)
        ):
            metrics = metrics_utils.compute_binary_classification_metrics(
                y_true=all_labels,
                y_prob=all_probs,
                threshold=self.cfg.threshold,
                pos_label=1,
            )
            metrics_utils.log_metrics(metrics, prefix="Eval")
            return metrics

        logger.info(
            "Evaluation finished without metric computation "
            "(compute_metrics=%s, has_labels=%s, has_any_labels=%s).",
            self.cfg.compute_metrics,
            self.cfg.has_labels,
            has_any_labels,
        )
        return None

    # --------------------------------------------------------
    # 内部工具：batch 处理
    # --------------------------------------------------------
    def _extract_ids(
        self,
        batch: Mapping[str, Any],
    ) -> Optional[Union[Tensor, Iterable[Any]]]:
        """
        @brief 从 batch 中抽取样本 id（支持键 'id' 或 'ids'）。
               Extract sample ids from batch (supports keys 'id' or 'ids').
        @param batch 单个 batch 的数据映射。Mapping for one batch.
        @return 返回 id 张量或序列；若未找到则返回 None。
                Returns id tensor/sequence, or None if not found.
        """
        if "id" in batch:
            return batch["id"]
        if "ids" in batch:
            return batch["ids"]
        return None

    def _extract_labels(
        self,
        batch: Mapping[str, Any],
    ) -> Optional[Union[Tensor, Iterable[int]]]:
        """
        @brief 从 batch 中抽取标签（键 'labels'）。
               Extract labels from batch (key 'labels').
        @param batch 单个 batch 的数据映射。Mapping for one batch.
        @return 返回标签张量或序列；若未找到则返回 None。
                Returns label tensor/sequence, or None if not found.
        """
        return batch.get("labels", None)

    def _forward_batch(self, batch: Mapping[str, Any]) -> Tensor:
        """
        @brief 对单个 batch 执行前向传播，返回 logits。
               Run forward pass for one batch and return logits.
        @param batch 单个 batch 数据映射。Mapping for one batch.
        @return logits 张量（展平成一维向量）。Logits tensor (flattened to 1D).
        @note 会自动将张量迁移到目标设备；键 'labels'/'id'/'ids' 不会传给模型。
              Tensors are moved to device; 'labels'/'id'/'ids' are not passed to model.
        """
        inputs: dict[str, Any] = {}

        for k, v in batch.items():
            if k in ("labels", "id", "ids"):
                continue
            if isinstance(v, Tensor):
                v = v.to(self.device)
            inputs[k] = v

        outputs = self.model(**inputs)

        if isinstance(outputs, Mapping):
            if "logits" in outputs:
                logits = outputs["logits"]
            else:
                logits = next(
                    (v for v in outputs.values() if isinstance(v, Tensor)),
                    None,
                )
                if logits is None:
                    raise RuntimeError(
                        "Model output mapping does not contain 'logits' "
                        "nor any Tensor values."
                    )
        elif isinstance(outputs, Tensor):
            logits = outputs
        else:
            raise RuntimeError(
                f"Unsupported model output type: {type(outputs)!r}, "
                "expected Tensor or Mapping."
            )

        logits = logits.view(-1)
        return logits
