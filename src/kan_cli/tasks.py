"""
@file tasks.py
@brief KAN CLI 具体任务实现：训练 / 评估 / 预测等。
       Concrete task implementations for the KAN CLI: train / evaluate / predict.
"""

from __future__ import annotations

from dataclasses import asdict
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F  # noqa: F401  # 预留给未来可能的扩展
from torch.utils.data import DataLoader

import kan
from kan import get_logger
from kan.repr.batching import BatchEncoding

from kan_cli.runtime import (
    ExperimentRuntime,
    ExperimentTask,
    register_task,
)

logger = get_logger(__name__)


# ============================================================
# 通用小工具：设备搬运 & Batch 解析
# Common helpers: device movement & batch parsing
# ============================================================


def _move_to_device(obj: Any, device: torch.device) -> Any:
    """
    @brief 将张量/嵌套结构移动到指定设备上。
           Move tensors / nested structures to the given device.
    @param obj 任意 Python 对象（张量、dict、list 等）。
           Arbitrary Python object (tensor, dict, list, etc.).
    @param device 目标设备。Target device.
    @return 移动后的对象。Object placed on the device.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [_move_to_device(v, device) for v in obj]
        return type(obj)(t)
    return obj


def _from_batch_encoding(
    batch: BatchEncoding,
    device: torch.device,
) -> Tuple[BatchEncoding, Optional[torch.Tensor], Optional[List[Any]]]:
    """
    @brief 从 BatchEncoding 构造“已搬到设备”的 BatchEncoding，并抽取标签与 ID。
           Build a device-placed BatchEncoding from the input, and extract labels and IDs.
    @param batch 原始 BatchEncoding 实例。Original BatchEncoding.
    @param device 目标设备。Target device.
    @return (batch_on_device, labels, ids_list) 三元组。
            A triple (batch_on_device, labels, ids_list).
    @note
        * labels 将被转换为 float32 并搬到 device，便于 BCEWithLogitsLoss 使用。
        * ids_list 始终是 CPU 上的 Python list，便于写 CSV。
    """
    # ---- 1) 搬运各字段到设备 Move tensors to device ----
    token_ids = batch.token_ids.to(device)
    token_padding_mask = batch.token_padding_mask.to(device)

    entity_ids = batch.entity_ids.to(device) if batch.entity_ids is not None else None
    entity_padding_mask = (
        batch.entity_padding_mask.to(device)
        if batch.entity_padding_mask is not None
        else None
    )
    context_ids = (
        batch.context_ids.to(device) if batch.context_ids is not None else None
    )
    context_padding_mask = (
        batch.context_padding_mask.to(device)
        if batch.context_padding_mask is not None
        else None
    )

    labels: Optional[torch.Tensor] = batch.labels  # type: ignore[assignment]
    if labels is not None:
        if not isinstance(labels, torch.Tensor):
            labels = torch.as_tensor(labels, dtype=torch.float32, device=device)
        else:
            labels = labels.to(device=device, dtype=torch.float32)

    ids_tensor = getattr(batch, "ids", None)
    ids_list: Optional[List[Any]] = None
    ids_on_device: Optional[torch.Tensor] = None

    if ids_tensor is not None:
        if isinstance(ids_tensor, torch.Tensor):
            ids_on_device = ids_tensor.to(device)
            ids_list = ids_tensor.detach().cpu().tolist()
        else:
            # 若 ids 不是 Tensor，则仅保留 Python list 形式
            ids_list = list(ids_tensor)
            ids_on_device = None

    batch_on_device = BatchEncoding(
        token_ids=token_ids,
        token_padding_mask=token_padding_mask,
        entity_ids=entity_ids,
        entity_padding_mask=entity_padding_mask,
        context_ids=context_ids,
        context_padding_mask=context_padding_mask,
        labels=labels,
        ids=ids_on_device,
    )

    return batch_on_device, labels, ids_list


def _extract_batch_io(
    batcher: Any,
    raw_batch: Any,
    device: torch.device,
) -> Tuple[Any, Optional[Tensor], Optional[List[Any]]]:
    """
    @brief 从原始 batch 中抽取“模型输入对象”、标签和样本 ID。
           Extract model input object, labels and sample IDs from a raw batch.
    @param batcher Batcher 实例，目前主要用于与 DataLoader 集成以及从
                   PreprocessedSample 序列构造 BatchEncoding。
           Batcher instance, mainly used as DataLoader collate_fn and to build
           BatchEncoding from a sequence of PreprocessedSample.
    @param raw_batch DataLoader 产出的原始 batch；当前项目中要么为
                     BatchEncoding，要么为 List[PreprocessedSample]。
           Raw batch yielded by the DataLoader; in this project it is either a
           BatchEncoding or a List[PreprocessedSample].
    @param device 目标设备。Target device.
    @return (model_inputs, labels, ids) 三元组；labels/ids 可能为 None。
            A triple (model_inputs, labels, ids); labels/ids may be None.

    @note
        解析优先级（从高到低）如下：
        The interpretation priority is:

        1) 若 raw_batch 是 BatchEncoding，直接构造 device 版 BatchEncoding。
           If raw_batch is a BatchEncoding, move it to device directly.
        2) 若 raw_batch 是 List[PreprocessedSample]，通过 batcher(raw_batch)
           构造 BatchEncoding 再搬到 device。
           If raw_batch is a List[PreprocessedSample], call batcher(raw_batch)
           to build a BatchEncoding and then move it to device.
        3) 其他情况：保留一个轻量回退逻辑（dict + batcher.to_model_inputs）以
           兼容未来扩展或旧代码。
           Otherwise: use a light fallback (dict + batcher.to_model_inputs) for
           extensibility / legacy code.
    """
    # ============================================================
    # 1) 正常路径：raw_batch 已经是 BatchEncoding
    #    Normal path: raw_batch is already a BatchEncoding.
    # ============================================================
    if isinstance(raw_batch, BatchEncoding):
        return _from_batch_encoding(raw_batch, device)

    # ============================================================
    # 1.5) 当前项目的实际情况：raw_batch 是 List[PreprocessedSample]
    #      DataLoader 使用 identity_collate，真正的 batching 由 Batcher 完成。
    #      In this project, raw_batch is List[PreprocessedSample] because
    #      DataLoader uses identity_collate and Batcher does the batching.
    # ============================================================
    PreprocessedSample = kan.PreprocessedSample
    if isinstance(raw_batch, (list, tuple)) and raw_batch:
        first = raw_batch[0]
        if isinstance(first, PreprocessedSample):
            batch_encoding = batcher(raw_batch)
            if not isinstance(batch_encoding, BatchEncoding):
                raise TypeError(
                    "Batcher(collate) is expected to return BatchEncoding, "
                    f"got {type(batch_encoding)!r}"
                )
            return _from_batch_encoding(batch_encoding, device)

    # ============================================================
    # 2) 回退逻辑：兼容非 BatchEncoding 的情况（例如未来扩展）
    #    Fallback logic for non-BatchEncoding batches.
    # ============================================================
    # 2.1 尝试 batcher 提供的通用接口（若存在）
    #     Try generic methods on batcher if available.
    if hasattr(batcher, "to_model_inputs"):
        model_inputs: Any = batcher.to_model_inputs(raw_batch)
    else:
        model_inputs = raw_batch

    labels: Optional[Tensor] = None
    ids: Optional[List[Any]] = None

    if hasattr(batcher, "get_labels"):
        labels = batcher.get_labels(raw_batch)  # type: ignore[assignment]
    if hasattr(batcher, "get_ids"):
        got_ids = batcher.get_ids(raw_batch)  # type: ignore[assignment]
        if got_ids is not None:
            ids = list(got_ids)

    # 2.2 若 labels/ids 仍为空，对 dict 做温和兜底
    #     Gentle fallback for dict-shaped batches.
    if labels is None and isinstance(raw_batch, dict):
        for key in ("labels", "label", "y"):
            if key in raw_batch:
                labels = raw_batch[key]  # type: ignore[assignment]
                break

    if ids is None and isinstance(raw_batch, dict):
        if "ids" in raw_batch:
            ids = list(raw_batch["ids"])
        elif "id" in raw_batch:
            ids = list(raw_batch["id"])

    # 2.3 统一搬到设备，并规范 labels 类型
    #     Normalize device and label dtype.
    model_inputs = _move_to_device(model_inputs, device)
    if labels is not None and not isinstance(labels, Tensor):
        labels = torch.as_tensor(labels, dtype=torch.float32, device=device)
    elif isinstance(labels, Tensor):
        labels = labels.to(device=device, dtype=torch.float32)

    return model_inputs, labels, ids


def _ensure_logits_tensor(
    outputs: Any,
    logger_obj: Any,
) -> Tensor:
    """
    @brief 从模型输出中抽取 logits 张量，兼容多种返回形式。
           Extract logits Tensor from model outputs, supporting multiple formats.
    @param outputs 模型原始输出，可以是 Tensor / tuple / dict。
           Raw outputs from the model: Tensor / tuple / dict.
    @param logger_obj 日志记录器，用于输出调试信息。
           Logger instance used for debugging information.
    @return logits 张量。Logits tensor.
    @throws RuntimeError 若无法从输出中解析出 logits。
            Raises RuntimeError if logits cannot be extracted.
    @note
        支持的形式包括：
        Supported formats include:

        1) Tensor 直接作为 logits。
           Tensor used directly as logits.
        2) (logits, aux) 元组，取第 0 个元素。
           (logits, aux) tuple, take element 0 as logits.
        3) {"logits": Tensor, ...} 映射，从 "logits" 键中提取。
           Mapping with "logits" key containing a Tensor.
    """
    if isinstance(outputs, Tensor):
        return outputs

    if isinstance(outputs, tuple):
        if not outputs:
            raise RuntimeError("Model returned empty tuple, cannot get logits.")
        logits = outputs[0]
        if not isinstance(logits, Tensor):
            raise RuntimeError(
                "First element of model output tuple is not a Tensor; "
                f"got type {type(logits)!r}."
            )
        return logits

    if isinstance(outputs, dict):
        if "logits" in outputs and isinstance(outputs["logits"], Tensor):
            return outputs["logits"]
        raise RuntimeError(
            "Model output dict must contain a Tensor under key 'logits', "
            f"got keys: {list(outputs.keys())!r}."
        )

    raise RuntimeError(
        f"Unsupported model output type: {type(outputs)!r}; "
        "expected Tensor, tuple, or dict with a 'logits' Tensor."
    )


def _adapt_logits_shape(
    logits: torch.Tensor,
    target: torch.Tensor,
    logger_obj: Any,
) -> torch.Tensor:
    """
    @brief 将模型输出的 logits 调整为与 target 兼容的一维张量。
           Adapt model logits to a 1D tensor compatible with target.
    @param logits 模型原始输出。Raw model output tensor.
    @param target 目标标签张量。Target label tensor.
    @param logger_obj 日志记录器。Logger instance.
    @return 形状与 target 匹配的一维 logits。1D logits matching target shape.
    """
    if logits.shape == target.shape:
        return logits

    if logits.ndim == 2:
        if logits.size(1) == 1:
            return logits.squeeze(1)
        if logits.size(1) == 2:
            logger_obj.info(
                "[tasks] Adapting 2-class logits -> positive-class logits via [:,1]."
            )
            return logits[:, 1]

    # 兜底：尝试展平后截断/广播 Fallback: flatten then adjust by size
    flat = logits.view(-1)
    if flat.numel() == target.numel():
        return flat.view_as(target)

    raise RuntimeError(
        f"Incompatible logits shape {logits.shape} for target shape {target.shape}."
    )


def _prepare_binary_logits_for_predict(
    logits: torch.Tensor,
    logger_obj: Any,
) -> torch.Tensor:
    """
    @brief 为预测任务准备一维二分类 logits。
           Prepare 1D binary logits for prediction.
    @param logits 模型输出的原始 logits。Raw logits from model.
    @param logger_obj 日志记录器实例。Logger instance.
    @return 形状为 (B,) 的一维 logits。1D logits of shape (B,).
    """
    if logits.ndim == 1:
        return logits

    if logits.ndim == 2:
        if logits.size(1) == 1:
            return logits.squeeze(1)
        if logits.size(1) == 2:
            logger_obj.info(
                "[tasks] Predict: adapting (B,2) logits -> positive-class via [:,1]."
            )
            return logits[:, 1]

    # 兜底：直接展平 Fallback: flatten
    logger_obj.warning(
        "[tasks] Predict: unexpected logits shape %s, flattening.",
        tuple(logits.shape),
    )
    return logits.view(-1)


def _ensure_eval_loader(runtime: ExperimentRuntime) -> DataLoader:
    """
    @brief 从 runtime 中选择用于评估/预测的 DataLoader。
           Select a DataLoader from runtime for evaluation/prediction.
    @param runtime 实验运行时。Experiment runtime.
    @return 非空的 DataLoader。
            A non-None DataLoader.
    @throws RuntimeError 若找不到合适的 DataLoader。
            Raises RuntimeError if no suitable loader is available.
    """
    if runtime.val_loader is not None:
        return runtime.val_loader  # type: ignore[return-value]
    if runtime.test_loader is not None:
        return runtime.test_loader  # type: ignore[return-value]
    if runtime.train_loader is not None:
        logger.warning(
            "[tasks] No dedicated val/test loader; falling back to train_loader."
        )
        return runtime.train_loader  # type: ignore[return-value]
    raise RuntimeError("No DataLoader available for evaluation/prediction.")


# ============================================================
# 公共执行器：二分类推理工具
# Shared executor: binary classification inference helper
# ============================================================


class _BinaryClassificationRunner:
    """
    @brief KAN 二分类推理小助手，封装 eval / predict 通用逻辑。
           Small helper for KAN binary classification, encapsulating common
           logic for eval / predict.
    """

    def __init__(
        self,
        runtime: ExperimentRuntime,
        checkpoint: Optional[Path | str] = None,
    ) -> None:
        """
        @brief 根据 runtime 和 checkpoint 构建推理环境。
               Build inference environment from runtime and checkpoint.
        @param runtime 实验运行时。Experiment runtime.
        @param checkpoint 可选 checkpoint 路径，用于加载权重。
               Optional checkpoint path to load weights from.
        """
        self.runtime = runtime
        self.device = runtime.device
        self.batcher = runtime.get_batcher(build_if_missing=True)
        # 让 runtime 负责构建 + 加载权重
        self.model = runtime.get_model(checkpoint=checkpoint)
        self.model.eval()

    def run_evaluation(
        self,
        loader: DataLoader,
    ) -> Tuple[List[float], List[float], List[Any]]:
        """
        @brief 在有标签数据集上跑一轮评估推理。
               Run one pass over a labeled dataset for evaluation.
        @param loader DataLoader，元素为 BatchEncoding 或预处理样本批次。
               DataLoader yielding BatchEncoding or preprocessed batches.
        @return (labels, probs, ids) 三元组。
                A triple (labels, probs, ids).
        """
        all_ids: List[Any] = []
        all_labels: List[float] = []
        all_probs: List[float] = []

        with torch.no_grad():
            for raw_batch in loader:
                model_inputs, labels, ids = _extract_batch_io(
                    batcher=self.batcher,
                    raw_batch=raw_batch,
                    device=self.device,
                )
                if labels is None:
                    raise RuntimeError(
                        "Evaluation requires labels, but got labels=None."
                    )

                if isinstance(model_inputs, dict):
                    outputs = self.model(**model_inputs)  # type: ignore[arg-type]
                else:
                    outputs = self.model(model_inputs)  # type: ignore[arg-type]

                logits = _ensure_logits_tensor(outputs, self.runtime.logger)
                logits = _adapt_logits_shape(logits, labels, self.runtime.logger)
                probs = torch.sigmoid(logits).detach().cpu()

                all_probs.extend(probs.tolist())
                all_labels.extend(labels.detach().cpu().tolist())

                if ids is not None:
                    all_ids.extend(ids)
                else:
                    start = len(all_ids)
                    all_ids.extend(range(start, start + probs.size(0)))

        return all_labels, all_probs, all_ids

    def run_prediction(
        self,
        loader: DataLoader,
    ) -> Tuple[List[float], List[Any]]:
        """
        @brief 在无标签数据集上跑一次预测，输出概率与样本 ID。
               Run prediction on an unlabeled dataset and return probs & ids.
        @param loader DataLoader，元素为 BatchEncoding 或预处理样本批次。
               DataLoader yielding BatchEncoding or preprocessed batches.
        @return (probs, ids) 二元组。
                A pair (probs, ids).
        """
        all_ids: List[Any] = []
        all_probs: List[float] = []

        with torch.no_grad():
            for raw_batch in loader:
                model_inputs, _labels, ids = _extract_batch_io(
                    batcher=self.batcher,
                    raw_batch=raw_batch,
                    device=self.device,
                )

                if isinstance(model_inputs, dict):
                    outputs = self.model(**model_inputs)  # type: ignore[arg-type]
                else:
                    outputs = self.model(model_inputs)  # type: ignore[arg-type]

                logits = _ensure_logits_tensor(outputs, self.runtime.logger)
                logits_1d = _prepare_binary_logits_for_predict(
                    logits, self.runtime.logger
                )
                probs = torch.sigmoid(logits_1d).detach().cpu()

                all_probs.extend(probs.tolist())

                if ids is not None:
                    all_ids.extend(ids)
                else:
                    start = len(all_ids)
                    all_ids.extend(range(start, start + probs.size(0)))

        return all_probs, all_ids


# ============================================================
# 训练任务 Train Task
# ============================================================


@register_task("train")
class TrainTask(ExperimentTask):
    """
    @brief 训练任务：从零开始训练一个 KAN 模型并保存 checkpoint。
           Training task: train a KAN model from scratch and save a checkpoint.
    """

    task_name: str = "train"

    def run(self) -> Path:
        """
        @brief 执行完整训练流程。
               Execute the full training loop.
        @return 已保存模型 checkpoint 的路径。
                Path to the saved model checkpoint.
        """
        runtime = self.runtime
        cfg = runtime.config
        training_cfg = getattr(cfg, "training", None)
        if training_cfg is None:
            raise ValueError("ExperimentConfig.training is required for training.")

        num_epochs: int = getattr(training_cfg, "num_epochs", 1)
        grad_clip: float = getattr(training_cfg, "gradient_clip", 0.0)

        train_loader = runtime.train_loader
        if train_loader is None:
            raise RuntimeError("TrainTask requires a non-empty train_loader.")

        # 懒加载 batcher / 模型 / 优化器 / 调度器
        batcher = runtime.get_batcher(build_if_missing=True)
        model = runtime.get_model()
        optimizer = runtime.get_optimizer()
        scheduler = runtime.get_scheduler()

        runtime.logger.info(
            "Start training: epochs=%d, grad_clip=%.3f, device=%s",
            num_epochs,
            grad_clip,
            runtime.device,
        )

        criterion = nn.BCEWithLogitsLoss()

        global_step = 0
        for epoch in range(1, num_epochs + 1):
            model.train()
            epoch_loss = 0.0
            epoch_samples = 0

            for raw_batch in train_loader:
                model_inputs, labels, _ = _extract_batch_io(
                    batcher=batcher,
                    raw_batch=raw_batch,
                    device=runtime.device,
                )
                if labels is None:
                    raise RuntimeError(
                        "TrainTask expects labels in each batch, but got None."
                    )

                # 支持两种调用方式：dict（旧风格）与 BatchEncoding（新风格）
                if isinstance(model_inputs, dict):
                    outputs = model(**model_inputs)  # type: ignore[arg-type]
                else:
                    outputs = model(model_inputs)  # type: ignore[arg-type]

                logits = _ensure_logits_tensor(outputs, runtime.logger)
                logits = _adapt_logits_shape(logits, labels, runtime.logger)

                loss = criterion(logits, labels)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                if grad_clip and grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                batch_size = labels.size(0)
                epoch_loss += float(loss.item()) * batch_size
                epoch_samples += batch_size
                global_step += 1

            avg_loss = epoch_loss / max(1, epoch_samples)
            runtime.logger.info(
                "Epoch %d/%d finished: avg_loss=%.6f, samples=%d",
                epoch,
                num_epochs,
                avg_loss,
                epoch_samples,
            )

        # ---------- 保存 checkpoint Save checkpoint ----------
        out_dir = Path(getattr(training_cfg, "output_dir", "train/models"))
        if not out_dir.is_absolute():
            out_dir = runtime.work_dir / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = out_dir / "model.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "config": asdict(training_cfg),
            },
            ckpt_path,
        )
        runtime.logger.info(
            "Training completed, checkpoint saved to %s",
            ckpt_path.as_posix(),
        )
        return ckpt_path


# ============================================================
# 评估任务 Evaluate Task
# ============================================================


@register_task("evaluate")
class EvaluateTask(ExperimentTask):
    """
    @brief 评估任务：加载已训练模型，在有标签数据集上计算指标。
           Evaluation task: load a trained model and compute metrics on a labeled dataset.
    """

    task_name: str = "evaluate"

    def run(self) -> Dict[str, Any]:
        """
        @brief 执行评估流程，返回指标并按需写入文件。
               Run evaluation and return metrics; optionally write them to files.
        @return 指标字典（如 accuracy / precision / recall / f1 等）。
                Metrics dictionary (e.g., accuracy / precision / recall / f1).
        """
        runtime = self.runtime
        params = self.params

        checkpoint_path = Path(params["checkpoint_path"])
        metrics_path: Optional[str] = params.get("metrics_path")
        probs_path: Optional[str] = params.get("probs_path")

        eval_loader = _ensure_eval_loader(runtime)

        runner = _BinaryClassificationRunner(runtime, checkpoint=checkpoint_path)
        all_labels, all_probs, all_ids = runner.run_evaluation(eval_loader)

        if not all_labels:
            raise RuntimeError("No samples collected during evaluation.")

        labels_tensor = torch.tensor(all_labels, dtype=torch.float32)
        probs_tensor = torch.tensor(all_probs, dtype=torch.float32)
        preds_tensor = (probs_tensor >= 0.5).to(torch.int64)
        labels_int = labels_tensor.to(torch.int64)

        correct = (preds_tensor == labels_int).sum().item()
        total = labels_int.numel()
        accuracy = correct / max(1, total)

        tp = int(((preds_tensor == 1) & (labels_int == 1)).sum().item())
        fp = int(((preds_tensor == 1) & (labels_int == 0)).sum().item())
        fn = int(((preds_tensor == 0) & (labels_int == 1)).sum().item())

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        metrics: Dict[str, Any] = {
            "num_samples": total,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        runtime.logger.info("[eval] metrics: %s", metrics)

        # ---------- 写出 metrics JSON ----------
        if metrics_path is not None:
            mpath = Path(metrics_path)
            if not mpath.is_absolute():
                mpath = runtime.work_dir / mpath
            mpath.parent.mkdir(parents=True, exist_ok=True)
            with mpath.open("w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            runtime.logger.info("[eval] Metrics written to %s", mpath.as_posix())

        # ---------- 写出概率 CSV ----------
        if probs_path is not None:
            ppath = Path(probs_path)
            if not ppath.is_absolute():
                ppath = runtime.work_dir / ppath
            ppath.parent.mkdir(parents=True, exist_ok=True)
            with ppath.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "prob", "label"])
                for i, p, y in zip(all_ids, all_probs, all_labels):
                    writer.writerow([i, p, int(y)])
            runtime.logger.info("[eval] Probabilities written to %s", ppath.as_posix())

        return metrics


# ============================================================
# 预测任务 Predict Task
# ============================================================


@register_task("predict")
class PredictTask(ExperimentTask):
    """
    @brief 预测任务：加载已训练模型，在无标签数据集上输出 id,prob。
           Prediction task: load a trained model and output id,prob on an unlabeled dataset.
    """

    task_name: str = "predict"

    def run(self) -> Path:
        """
        @brief 执行预测任务，将结果写入 CSV，并返回路径。
               Run prediction, write CSV results, and return the path.
        @return 预测结果 CSV 文件路径。
                Path to the prediction CSV file.
        """
        runtime = self.runtime
        params = self.params

        checkpoint_path = Path(params["checkpoint_path"])
        output_path = Path(params.get("output_path", "results.csv"))

        pred_loader = _ensure_eval_loader(runtime)
        runner = _BinaryClassificationRunner(runtime, checkpoint=checkpoint_path)
        all_probs, all_ids = runner.run_prediction(pred_loader)

        if not output_path.is_absolute():
            output_path = runtime.work_dir / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "prob"])
            for i, p in zip(all_ids, all_probs):
                writer.writerow([i, p])

        runtime.logger.info(
            "[predict] Predictions written to %s",
            output_path.as_posix(),
        )
        return output_path
