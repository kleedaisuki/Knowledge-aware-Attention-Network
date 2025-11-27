"""
@file trainer.py
@brief 训练流水线实现：根据 TrainingConfig 进行纯训练，不做评估。
       Training pipeline: run pure training according to TrainingConfig, no evaluation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Iterable

import torch
from torch import nn, Tensor
from torch.optim import AdamW

from kan.utils.logging import get_logger
from kan.repr.batching import BatchEncoding

logger = get_logger(__name__)


# ============================================================
# 配置定义 TrainingConfig
# ============================================================


@dataclass
class TrainingConfig:
    """
    @brief 训练相关超参数配置。Training hyper-parameters configuration.
    @param num_epochs 训练轮数。Number of training epochs.
    @param learning_rate 学习率。Learning rate.
    @param weight_decay 权重衰减系数。Weight decay factor.
    @param warmup_steps 线性 warmup 的步数（0 表示不启用）。Steps for linear warmup (0 disables).
    @param gradient_clip 梯度裁剪阈值（0 表示不裁剪）。Max grad-norm for clipping (0 disables).

    @param device 训练设备标识，例如 "cuda" 或 "cpu"。
                  Device string, e.g. "cuda" or "cpu".
    @param seed 全局随机种子。Global random seed.

    @param output_dir 模型保存目录。Directory to save model checkpoints.
    @param log_dir 日志目录（当前主要用于占位，日志通过 logging 模块输出）。
                   Log directory (currently informational; logging uses logging module).
    @param save_every 每多少个 epoch 保存一次模型。Save checkpoint every N epochs.
    """

    num_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    warmup_steps: int = 0
    gradient_clip: float = 0.0

    device: str = "cuda"
    seed: int = 42

    output_dir: str = "train/models"
    log_dir: str = "train/logs"
    save_every: int = 1


# ============================================================
# 训练器 Trainer
# ============================================================


class Trainer:
    """
    @brief 纯训练用 Trainer：只负责前向 + 反向 + 保存权重，不做评估。
           Pure training Trainer: forward + backward + checkpoint saving, no evaluation.

    @note 本 Trainer 只支持 kan.repr.batching.BatchEncoding 这一种 batch 形式，
          推荐配合 Batcher 作为 DataLoader 的 collate_fn 使用。
          This Trainer only supports kan.repr.batching.BatchEncoding batches,
          typically produced by Batcher as DataLoader.collate_fn.
    """

    def __init__(
        self,
        model: nn.Module,
        train_data: Iterable[BatchEncoding],
        cfg: TrainingConfig,
    ) -> None:
        """
        @brief 构造 Trainer 实例。
               Construct a Trainer instance.
        @param model 待训练的 PyTorch 模型（例如 KAN）。Model to train (e.g. KAN).
        @param train_data 可迭代的训练 batch 序列，每个 batch 为 BatchEncoding。
               Iterable of training batches, each a BatchEncoding instance.
        @param cfg TrainingConfig 训练配置。Training configuration.
        """
        self.model = model
        self.train_data = train_data
        self.cfg = cfg

        self.device = self._prepare_device(cfg.device)
        self.model.to(self.device)

        self._set_seed(cfg.seed)

        self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        os.makedirs(self.cfg.output_dir, exist_ok=True)
        os.makedirs(self.cfg.log_dir, exist_ok=True)

        logger.info(
            "Trainer initialized: epochs=%d, lr=%.3e, weight_decay=%.2e, "
            "warmup_steps=%d, grad_clip=%.3f, device=%s",
            self.cfg.num_epochs,
            self.cfg.learning_rate,
            self.cfg.weight_decay,
            self.cfg.warmup_steps,
            self.cfg.gradient_clip,
            self.device.type,
        )

    # --------------------------------------------------------
    # 辅助工具：设备 & 随机种子
    # --------------------------------------------------------
    def _prepare_device(self, device_str: str) -> torch.device:
        """
        @brief 解析设备字符串并做 fallback（例如 cuda 不可用时退回 cpu）。
               Parse device string and fallback (e.g. to CPU if CUDA unavailable).
        @param device_str 配置中的 device 字符串。Device string from config.
        @return torch.device 对象。torch.device instance.
        """
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(
                "CUDA requested but not available, falling back to CPU. "
                "Requested device=%s",
                device_str,
            )
            return torch.device("cpu")
        return torch.device(device_str)

    def _set_seed(self, seed: int) -> None:
        """
        @brief 设置全局随机种子。
               Set global random seed.
        @param seed 随机种子值。Random seed.
        """
        from kan.utils import seed as seed_utils  # type: ignore[import-not-found]

        seed_utils.set_global_seed(seed)  # type: ignore[attr-defined]
        logger.info("Global seed set via kan.utils.seed: %d", seed)
        return

    # --------------------------------------------------------
    # Warmup 学习率调度（仅线性 warmup）
    # --------------------------------------------------------
    def _update_learning_rate(self, global_step: int) -> None:
        """
        @brief 依据 warmup_steps 更新学习率（线性从 0 → base_lr）。
               Update learning rate according to warmup_steps (linear 0 → base_lr).
        @param global_step 当前已经执行的优化步数（从 1 开始）。Current global step (1-based).
        @note 若 warmup_steps=0，则不做任何调整。
              If warmup_steps=0, does nothing.
        """
        if self.cfg.warmup_steps <= 0:
            return

        if global_step >= self.cfg.warmup_steps:
            lr = self.cfg.learning_rate
        else:
            # 线性 warmup：lr = base_lr * step / warmup_steps
            lr = (
                self.cfg.learning_rate
                * float(global_step)
                / float(max(1, self.cfg.warmup_steps))
            )

        for group in self.optimizer.param_groups:
            group["lr"] = lr

    # --------------------------------------------------------
    # 训练主循环 Training Loop
    # --------------------------------------------------------
    def train(self) -> None:
        """
        @brief 执行完整训练流程，完成后在 output_dir 中写入若干模型权重文件。
               Run the full training loop; writes model checkpoints to output_dir.
        @note 本函数不做任何验证/测试，只打印训练 loss 并保存模型。
              This function performs no validation/testing; only logs training
              loss and saves model weights.
        """
        self.model.train()
        global_step = 0

        for epoch in range(1, self.cfg.num_epochs + 1):
            epoch_loss = 0.0
            num_batches = 0

            for batch in self.train_data:
                global_step += 1
                self._update_learning_rate(global_step)

                loss = self._train_one_batch(batch)
                epoch_loss += loss
                num_batches += 1

            avg_loss = epoch_loss / max(1, num_batches)
            logger.info(
                "Epoch %d/%d finished - avg_loss=%.6f",
                epoch,
                self.cfg.num_epochs,
                avg_loss,
            )

            if self.cfg.save_every > 0 and (epoch % self.cfg.save_every == 0):
                self._save_checkpoint(epoch)

        logger.info("Training finished. Total steps: %d", global_step)

    # --------------------------------------------------------
    # 单 batch 训练（只接受 BatchEncoding）
    # --------------------------------------------------------
    def _train_one_batch(self, batch: BatchEncoding) -> float:
        """
        @brief 对单个 batch 执行前向 + 反向 + 更新参数。
               Run forward + backward + parameter update for one BatchEncoding.
        @param batch 一个 batch 的数据，必须是 BatchEncoding，并包含 labels。
               One training batch; must be BatchEncoding with non-None labels.
        @return 当前 batch 的标量 loss 值（Python float）。
                Scalar loss value for this batch (Python float).
        """
        if batch.labels is None:
            raise RuntimeError(
                "BatchEncoding.labels is None; training requires labels."
            )

        # ---------- 将 BatchEncoding 中的张量移动到目标设备 ----------
        inputs: dict[str, Tensor] = {}

        # 文本部分
        inputs["token_ids"] = batch.token_ids.to(self.device)
        inputs["token_padding_mask"] = batch.token_padding_mask.to(self.device)

        # 实体 & 上下文部分（可能为空）
        if batch.entity_ids is not None:
            inputs["entity_ids"] = batch.entity_ids.to(self.device)
        if batch.entity_padding_mask is not None:
            inputs["entity_padding_mask"] = batch.entity_padding_mask.to(self.device)
        if batch.context_ids is not None:
            inputs["context_ids"] = batch.context_ids.to(self.device)
        if batch.context_padding_mask is not None:
            inputs["context_padding_mask"] = batch.context_padding_mask.to(self.device)

        labels: Tensor = batch.labels.to(self.device)

        # ---------- 前向 + 反向 + 更新 ----------
        self.optimizer.zero_grad(set_to_none=True)

        # 模型输出支持 Tensor 或 Mapping(logits=...)
        outputs = self.model(**inputs)

        if isinstance(outputs, dict):
            if "logits" in outputs:
                logits = outputs["logits"]
            else:
                # 若模型返回 dict 但不含 logits，则尝试取第一个 Tensor 项
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

        # BCEWithLogitsLoss 期望 float 目标
        labels_float = labels.float().view(-1)
        logits_flat = logits.view(-1)

        if logits_flat.shape != labels_float.shape:
            raise RuntimeError(
                f"Logits and labels shape mismatch: {tuple(logits_flat.shape)} "
                f"vs {tuple(labels_float.shape)}"
            )

        loss = self.criterion(logits_flat, labels_float)
        loss.backward()

        # 可选梯度裁剪
        if self.cfg.gradient_clip > 0.0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clip)

        self.optimizer.step()

        return float(loss.item())

    # --------------------------------------------------------
    # 模型保存 Checkpoint Saving
    # --------------------------------------------------------
    def _save_checkpoint(self, epoch: int) -> None:
        """
        @brief 保存当前模型权重到指定目录。
               Save current model checkpoint to the configured output directory.
        @param epoch 当前 epoch 序号，用于生成文件名。Current epoch index, used in filename.
        """
        ckpt_path = os.path.join(self.cfg.output_dir, f"epoch{epoch}.pt")

        # 只保存必要信息：模型参数 + 训练配置
        state = {
            "model_state_dict": self.model.state_dict(),
            "training_config": asdict(self.cfg),
            "epoch": epoch,
        }

        try:
            torch.save(state, ckpt_path)
            logger.info("Checkpoint saved: %s", ckpt_path)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to save checkpoint to %s: %s", ckpt_path, e)
