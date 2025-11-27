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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F  # noqa: F401  # 预留给未来可能的扩展
from torch.utils.data import DataLoader

from kan import get_logger

from kan_cli.runtime import (
    ExperimentRuntime,
    ExperimentTask,
    RuntimeState,  # noqa: F401  # 目前未直接使用，保留作类型语义
    register_task,
)


logger = get_logger(__name__)


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


def _extract_batch_io(
    batcher: Any,
    raw_batch: Any,
    device: torch.device,
) -> Tuple[Dict[str, Any], Optional[torch.Tensor], Optional[List[Any]]]:
    """
    @brief 从原始 batch 中抽取模型输入、标签和样本 ID。
           Extract model inputs, labels and sample IDs from a raw batch.
    @param batcher Batcher 实例，用于自定义拆解逻辑。
           Batcher instance, allowed to customize how batch is interpreted.
    @param raw_batch DataLoader 产出的原始 batch。
           Raw batch object yielded by the DataLoader.
    @param device 目标设备。Target device.
    @return (model_inputs, labels, ids) 三元组；labels/ids 可能为 None。
            A triple (model_inputs, labels, ids); labels/ids may be None.

    @note
        为了尽量降低对 Batcher 实现细节的假设，本函数按以下优先级解析 batch：
        To minimize reliance on Batcher internals, the function interprets batch
        using the following priority:

        1) 若 batcher 实现了 to_model_inputs / get_labels / get_ids，则优先使用。
           If batcher implements to_model_inputs / get_labels / get_ids, use them.
        2) 否则，若 raw_batch 是 dict，则：
           Otherwise, if raw_batch is a dict:
           - model_inputs 取 raw_batch["model_inputs"] 或除 label/id 外的键；
             model_inputs = raw_batch["model_inputs"] or all keys except label/id.
           - labels 尝试查找 "labels"/"label"/"y"。
           - ids 尝试查找 "ids"/"id"。
        3) 否则，若 raw_batch 是 (inputs, labels[, ids]) 元组/列表，则按位置取。
           Otherwise, if raw_batch is a tuple/list (inputs, labels[, ids]), use indices.
    """
    # 1) Batcher 显式方法 Explicit methods on batcher
    if hasattr(batcher, "to_model_inputs"):
        model_inputs = batcher.to_model_inputs(raw_batch)
    else:
        model_inputs = None

    labels: Optional[torch.Tensor] = None
    ids: Optional[List[Any]] = None

    if hasattr(batcher, "get_labels"):
        labels = batcher.get_labels(raw_batch)  # type: ignore[assignment]
    if hasattr(batcher, "get_ids"):
        ids = list(batcher.get_ids(raw_batch))  # type: ignore[arg-type]

    # 2) 基于 dict 结构的回退 Fallback for dict-like batches
    if model_inputs is None:
        if isinstance(raw_batch, dict):
            if "model_inputs" in raw_batch:
                model_inputs = raw_batch["model_inputs"]
            else:
                # 过滤掉 label/id 字段
                exclude = {"label", "labels", "y", "id", "ids"}
                model_inputs = {k: v for k, v in raw_batch.items() if k not in exclude}
        elif isinstance(raw_batch, (list, tuple)) and raw_batch:
            # 3) (inputs, labels[, ids]) 风格
            model_inputs = raw_batch[0]
        else:
            # 最后兜底就直接当成输入
            model_inputs = raw_batch

    if labels is None:
        if isinstance(raw_batch, dict):
            for key in ("labels", "label", "y"):
                if key in raw_batch:
                    labels = raw_batch[key]
                    break
        elif isinstance(raw_batch, (list, tuple)) and len(raw_batch) >= 2:
            labels = raw_batch[1]

    if ids is None:
        if isinstance(raw_batch, dict):
            if "ids" in raw_batch:
                ids = list(raw_batch["ids"])
            elif "id" in raw_batch:
                ids = list(raw_batch["id"])
        elif isinstance(raw_batch, (list, tuple)) and len(raw_batch) >= 3:
            ids = list(raw_batch[2])

    # 统一张量类型 & 设备 Normalize tensors & device
    model_inputs = _move_to_device(model_inputs, device)
    if labels is not None and not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels, dtype=torch.float32, device=device)
    elif isinstance(labels, torch.Tensor):
        labels = labels.to(device=device, dtype=torch.float32)

    return model_inputs, labels, ids


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

        batcher = runtime.get_batcher(build_if_missing=True)
        model = runtime.get_model()
        optimizer = runtime.get_optimizer()
        scheduler = runtime.get_scheduler()

        runtime.logger.info(
            "[train] Start training: epochs=%d, grad_clip=%.3f, device=%s",
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

                if isinstance(model_inputs, dict):
                    logits = model(**model_inputs)  # type: ignore[arg-type]
                else:
                    logits = model(model_inputs)  # type: ignore[arg-type]

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
                "[train] Epoch %d/%d finished: avg_loss=%.6f, samples=%d",
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
            "[train] Training completed, checkpoint saved to %s",
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
        batcher = runtime.get_batcher(build_if_missing=True)

        model = runtime.get_model(checkpoint=checkpoint_path)
        model.eval()

        all_ids: List[Any] = []
        all_labels: List[float] = []
        all_probs: List[float] = []

        with torch.no_grad():
            for raw_batch in eval_loader:
                model_inputs, labels, ids = _extract_batch_io(
                    batcher=batcher,
                    raw_batch=raw_batch,
                    device=runtime.device,
                )
                if labels is None:
                    raise RuntimeError(
                        "EvaluateTask requires ground-truth labels, but labels is None."
                    )

                if isinstance(model_inputs, dict):
                    logits = model(**model_inputs)  # type: ignore[arg-type]
                else:
                    logits = model(model_inputs)  # type: ignore[arg-type]

                logits = _adapt_logits_shape(logits, labels, runtime.logger)
                probs = torch.sigmoid(logits).detach().cpu()

                all_probs.extend(probs.tolist())
                all_labels.extend(labels.detach().cpu().tolist())

                if ids is not None:
                    all_ids.extend(ids)
                else:
                    # 若无显式 ID，则使用递增整数。
                    start = len(all_ids)
                    all_ids.extend(range(start, start + probs.size(0)))

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
        batcher = runtime.get_batcher(build_if_missing=True)

        model = runtime.get_model(checkpoint=checkpoint_path)
        model.eval()

        all_ids: List[Any] = []
        all_probs: List[float] = []

        with torch.no_grad():
            for raw_batch in pred_loader:
                model_inputs, _labels, ids = _extract_batch_io(
                    batcher=batcher,
                    raw_batch=raw_batch,
                    device=runtime.device,
                )

                if isinstance(model_inputs, dict):
                    logits = model(**model_inputs)  # type: ignore[arg-type]
                else:
                    logits = model(model_inputs)  # type: ignore[arg-type]

                logits = logits.view(-1)
                probs = torch.sigmoid(logits).detach().cpu()

                all_probs.extend(probs.tolist())

                if ids is not None:
                    all_ids.extend(ids)
                else:
                    start = len(all_ids)
                    all_ids.extend(range(start, start + probs.size(0)))

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
