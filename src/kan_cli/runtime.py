"""
@file runtime.py
@brief 统一实验运行时环境与任务调度（状态机 + 任务注册表）。
       Unified experiment runtime environment and task scheduling
       (state machine + task registry).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
from typing import Any, Callable, ClassVar, Dict, List, Optional, Type, TypeVar

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from kan import ExperimentConfig
from kan import get_logger
from kan import set_global_seed

from kan_cli.helpers import (
    build_all_dataloaders,
    build_or_load_vocabs,
    build_batcher,
    build_model_from_config,
    build_optimizer,
    build_scheduler,
)


# ============================================================
# 状态机定义 Runtime State Machine
# ============================================================


class RuntimeState(Enum):
    """
    @brief 运行时状态枚举：描述 ExperimentRuntime 的生命周期阶段。
           Runtime state enum: describes ExperimentRuntime lifecycle stages.
    """

    INIT = "init"  # 尚未就绪，初始化中 Not ready yet, initializing
    READY = "ready"  # 环境已就绪，可启动任务 Ready, tasks can start
    RUNNING = "running"  # 正在执行某个任务 A task is currently running
    COMPLETED = "completed"  # 最近一次任务成功完成 Last task completed successfully
    FAILED = "failed"  # 最近一次任务失败 Last task failed


# 任务名称类型别名 Task name alias
TaskName = str


# 前向声明 Forward reference for type hints
class ExperimentRuntime:  # noqa: D401 - doc in class below
    """占位以供类型提示使用 Placeholder for type hints."""

    ...


# ============================================================
# 任务基类与注册表 Task Base Class & Registry
# ============================================================

TTask = TypeVar("TTask", bound="ExperimentTask")


class ExperimentTask:
    """
    @brief 任务基类：所有具体任务（训练/评估/预测等）应继承自本类。
           Base class for tasks: all concrete tasks (train / eval / predict / etc.)
           should inherit from this class.

    @note  任务通过访问 self.runtime 来使用运行时环境（模型/数据/配置等）。
           Tasks access the runtime environment (model/data/config/etc.) via
           self.runtime.
    """

    #: @brief 任务名称（在注册表中的 key），子类必须覆盖。
    #:        Task name (key in registry); subclasses must override.
    task_name: ClassVar[TaskName] = "base"

    #: @brief 允许任务启动时的运行时状态集合。
    #:        Allowed runtime states at task start.
    allowed_start_states: ClassVar[List[RuntimeState]] = [
        RuntimeState.READY,
        RuntimeState.COMPLETED,
        RuntimeState.FAILED,
    ]

    def __init__(self, runtime: ExperimentRuntime, **kwargs: Any) -> None:
        """
        @brief 构造任务实例，持有运行时环境与任务参数。
               Construct a task instance, holding runtime and task parameters.

        @param runtime 运行时环境对象。Experiment runtime instance.
        @param kwargs  任务特定的额外参数。Task-specific extra parameters.
        """
        self.runtime = runtime
        self.params = kwargs

    def before_run(self) -> None:
        """
        @brief 任务开始前的钩子，可选重写。
               Optional hook executed before task.run().
        """
        # 默认不做任何事 Default: do nothing
        return

    def after_run(self, result: Any) -> None:
        """
        @brief 任务完成后的钩子，可选重写。
               Optional hook executed after task.run() returns.

        @param result 任务返回结果。Task return value.
        """
        # 默认不做任何事 Default: do nothing
        return

    def run(self) -> Any:  # pragma: no cover - abstract by convention
        """
        @brief 任务核心逻辑，子类必须实现。
               Core logic of the task; subclasses must implement.

        @return 任务执行结果，可为任意类型。
                Task execution result, arbitrary type.

        @note  建议实现时使用 runtime 提供的接口，如
               runtime.get_model() / runtime.get_dataloaders() 等。
               It is recommended to use runtime methods such as
               runtime.get_model() / runtime.get_dataloaders() etc.
        """
        raise NotImplementedError(
            "ExperimentTask.run() must be implemented by subclasses."
        )


class TaskRegistry:
    """
    @brief 任务注册表：维护任务名称到任务类的映射。
           Task registry: maintain mapping from task name to task class.

    @note  runtime 本身只与注册表交互，不直接依赖任何具体任务。
           The runtime interacts only with this registry, not concrete tasks.
    """

    def __init__(self) -> None:
        """@brief 初始化空注册表。Initialize an empty registry."""
        self._tasks: Dict[TaskName, Type[ExperimentTask]] = {}

    def register(
        self, task_cls: Type[TTask], name: Optional[TaskName] = None
    ) -> Type[TTask]:
        """
        @brief 注册一个任务类到注册表中。
               Register a task class into the registry.

        @param task_cls 要注册的任务类。Task class to register.
        @param name     注册使用的任务名称，若为 None 则使用 task_cls.task_name。
                        Task name to use; if None, use task_cls.task_name.

        @return 传入的 task_cls，便于作为装饰器使用。
                The same task_cls, convenient for decorator usage.

        @note  若名称冲突，将覆盖旧任务并给出告警日志。
               If the name already exists, the old task will be overwritten with a warning.
        """
        task_name = name or getattr(task_cls, "task_name", None)
        if not task_name:
            raise ValueError("Task class must define a non-empty 'task_name'.")

        if task_name in self._tasks:
            # 为了方便实验，允许覆盖，但记录告警
            logger = get_logger(__name__)
            logger.warning(f"Task '{task_name}' is already registered; overriding.")

        self._tasks[task_name] = task_cls
        return task_cls

    def get(self, name: TaskName) -> Type[ExperimentTask]:
        """
        @brief 根据名称获取任务类。
               Get a task class by name.

        @param name 任务名称。Task name.

        @return 对应的任务类。Corresponding task class.

        @throws KeyError 当任务未注册时抛出。
                Raises KeyError if the task is not registered.
        """
        if name not in self._tasks:
            raise KeyError(f"Task '{name}' is not registered.")
        return self._tasks[name]

    def available_tasks(self) -> List[TaskName]:
        """
        @brief 返回当前已注册的全部任务名称列表。
               Return the list of all registered task names.
        """
        return sorted(self._tasks.keys())


# 全局任务注册表 Global task registry
TASK_REGISTRY = TaskRegistry()


def register_task(
    name: Optional[TaskName] = None,
) -> Callable[[Type[TTask]], Type[TTask]]:
    """
    @brief 任务注册装饰器，供具体任务类使用。
           Decorator for task registration, used by concrete task classes.

    @param name 可选的任务名称，若为 None 则使用类的 task_name 字段。
                Optional task name; if None, use the class's task_name field.

    @example
        @register_task("train")
        class TrainTask(ExperimentTask):
            task_name = "train"
            ...
    """

    def decorator(cls: Type[TTask]) -> Type[TTask]:
        TASK_REGISTRY.register(cls, name=name)
        return cls

    return decorator


# ============================================================
# 运行时环境 ExperimentRuntime
# ============================================================


@dataclass
class ExperimentRuntime:
    """
    @brief 实验运行时环境，封装配置、设备、路径和懒加载资源接口。
           Experiment runtime environment, encapsulating configuration, device,
           paths, and lazy-loaded resource interfaces.

    @note  本类不包含任何特定任务逻辑，仅提供资源获取和状态管理。
           This class contains no task-specific logic; it only provides
           resource access and state management.
    """

    config: ExperimentConfig = field(repr=False)
    """@brief 实验配置。Experiment configuration."""

    work_dir: Path
    """@brief 实验工作目录（模型、日志、词表等的根目录）。
              Working directory for the experiment (models/logs/vocabs/etc.)."""

    device: torch.device
    """@brief 计算设备。Compute device."""

    logger: "logging.Logger"
    """@brief 日志记录器实例。Logger instance."""

    state: RuntimeState = field(default=RuntimeState.INIT)
    """@brief 当前运行时状态。Current runtime state."""

    # 内部缓存字段 Internal caches
    _train_loader: Optional[DataLoader] = field(default=None, init=False, repr=False)
    _val_loader: Optional[DataLoader] = field(default=None, init=False, repr=False)
    _test_loader: Optional[DataLoader] = field(default=None, init=False, repr=False)

    _text_vocab: Any = field(default=None, init=False, repr=False)
    _entity_vocab: Any = field(default=None, init=False, repr=False)

    _batcher: Any = field(default=None, init=False, repr=False)

    _model: Optional[nn.Module] = field(default=None, init=False, repr=False)
    _optimizer: Optional[Optimizer] = field(default=None, init=False, repr=False)
    _scheduler: Optional[_LRScheduler] = field(default=None, init=False, repr=False)

    def transition_to(self, new_state: RuntimeState) -> None:
        """
        @brief 切换运行时状态，带基本合法性检查。
               Transition runtime state with basic validation.

        @param new_state 目标状态。Target state.

        @note  当前实现仅做简单检查：禁止从 FAILED 直接跳到 RUNNING，
               其余场景由上层逻辑保证。
               Current implementation only prevents transitions like FAILED→RUNNING
               directly; other invariants should be maintained by callers.
        """
        if self.state == RuntimeState.FAILED and new_state == RuntimeState.RUNNING:
            raise RuntimeError("Cannot transition from FAILED to RUNNING directly.")
        self.logger.info(f"State: {self.state.value} -> {new_state.value}")
        self.state = new_state

    # ----------------------
    # 数据加载相关 DataLoaders
    # ----------------------

    def get_dataloaders(
        self,
    ) -> tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
        """
        @brief 懒加载并返回 (train, val, test) 三个 DataLoader。
               Lazily build and return (train, val, test) DataLoaders.

        @return 一个三元组 (train_loader, val_loader, test_loader)，可能包含 None。
                A triple (train_loader, val_loader, test_loader); elements may be None.

        @note  具体如何从 ExperimentConfig 构建 DataLoader 由
               build_all_dataloaders 来完成。
               The details of building DataLoaders from ExperimentConfig are
               delegated to build_all_dataloaders.
        """
        if (
            self._train_loader is None
            and self._val_loader is None
            and self._test_loader is None
        ):
            self.logger.info("Building all dataloaders...")
            train, val, test = build_all_dataloaders(self.config)
            self._train_loader, self._val_loader, self._test_loader = train, val, test
            self.logger.info("Dataloaders built.")
        return self._train_loader, self._val_loader, self._test_loader

    @property
    def train_loader(self) -> Optional[DataLoader]:
        """@brief 训练 DataLoader 属性访问。Training DataLoader property."""
        train, _, _ = self.get_dataloaders()
        return train

    @property
    def val_loader(self) -> Optional[DataLoader]:
        """@brief 验证 DataLoader 属性访问。Validation DataLoader property."""
        _, val, _ = self.get_dataloaders()
        return val

    @property
    def test_loader(self) -> Optional[DataLoader]:
        """@brief 测试/预测 DataLoader 属性访问。Test/predict DataLoader property."""
        _, _, test = self.get_dataloaders()
        return test

    # ----------------------
    # 词表与批处理 Vocabs & Batcher
    # ----------------------

    def get_vocabs(self, build_if_missing: bool = False) -> tuple[Any, Any]:
        """
        @brief 获取文本与实体词表，必要时可构建并持久化。
               Get text and entity vocabs; optionally build & persist them.

        @param build_if_missing 若为 True 且当前无词表，则从训练数据构建并保存。
                                If True and vocabs are missing, build from training
                                data and save them.

        @return (text_vocab, entity_vocab) 二元组。
                A pair (text_vocab, entity_vocab).

        @note  具体构建/加载逻辑由 build_or_load_vocabs 实现，建议接受
               (config, work_dir, train_loader, build_if_missing) 等参数。
               The actual build/load logic should be implemented in
               build_or_load_vocabs, which is recommended to accept
               (config, work_dir, train_loader, build_if_missing).
        """
        if self._text_vocab is None or self._entity_vocab is None:
            self.logger.info("Loading/building vocabs...")
            train_loader, _, _ = self.get_dataloaders()
            self._text_vocab, self._entity_vocab = build_or_load_vocabs(
                config=self.config,
                work_dir=self.work_dir,
                train_loader=train_loader,
                build_if_missing=build_if_missing,
            )
            self.logger.info("Vocabs ready.")
        return self._text_vocab, self._entity_vocab

    def get_batcher(self, build_if_missing: bool = True) -> Any:
        """
        @brief 获取批处理/张量化组件（Batcher）。
               Get the batcher component.

        @param build_if_missing 若为 True 且 batcher 为空，则构建之。
                                If True and batcher is missing, build it.

        @return Batcher 实例。Batcher instance.

        @note  实际构建逻辑由 build_batcher 实现，建议接受
               (config, text_vocab, entity_vocab)。
               Actual construction is delegated to build_batcher, which is
               recommended to accept (config, text_vocab, entity_vocab).
        """
        if self._batcher is None and build_if_missing:
            self.logger.info("Building batcher...")
            text_vocab, entity_vocab = self.get_vocabs(build_if_missing=True)
            self._batcher = build_batcher(
                config=self.config,
                text_vocab=text_vocab,
                entity_vocab=entity_vocab,
            )
            self.logger.info("Batcher built.")
        return self._batcher

    # ----------------------
    # 模型与优化器 Model & Optimizer
    # ----------------------

    def get_model(
        self, checkpoint: Optional[Path | str] = None, rebuild: bool = False
    ) -> nn.Module:
        """
        @brief 获取 KAN 模型，必要时构建并加载 checkpoint。
               Get KAN model; optionally build and load checkpoint.

        @param checkpoint 可选 checkpoint 路径，用于加载模型参数。
                          Optional checkpoint path for loading model state.
        @param rebuild    若为 True，则无论当前是否已有模型均重新构建。
                          If True, rebuild model even if it already exists.

        @return 模型实例（已迁移到 device 上）。
                Model instance (already moved to self.device).
        """
        if self._model is None or rebuild:
            self.logger.info("Building model from config...")
            self._model = build_model_from_config(self.config).to(self.device)
            self.logger.info("Model built.")

        if checkpoint is not None:
            ckpt_path = Path(checkpoint)
            if ckpt_path.is_file():
                self.logger.info(f"Loading checkpoint from {ckpt_path}...")
                state = torch.load(ckpt_path, map_location=self.device)
                # 这里假定 checkpoint 中 key 为 "model"
                self._model.load_state_dict(state["model"])
                self.logger.info("Checkpoint loaded.")
            else:
                self.logger.warning(f"Checkpoint not found: {ckpt_path}, skip loading.")

        return self._model

    def get_optimizer(self, rebuild: bool = False) -> Optimizer:
        """
        @brief 获取优化器，必要时构建。
               Get optimizer; build if necessary.

        @param rebuild 若为 True，则重新构建优化器。
                       If True, rebuild optimizer.

        @return 优化器实例。Optimizer instance.
        """
        if self._optimizer is None or rebuild:
            if self._model is None:
                raise RuntimeError("Model must be built before building optimizer.")
            self.logger.info("Building optimizer...")
            self._optimizer = build_optimizer(self.config, self._model)
            self.logger.info("Optimizer built.")
        return self._optimizer

    def get_scheduler(self, rebuild: bool = False) -> Optional[_LRScheduler]:
        """
        @brief 获取学习率调度器，必要时构建。
               Get learning rate scheduler; build if necessary.

        @param rebuild 若为 True，则重新构建调度器。
                       If True, rebuild scheduler.

        @return 调度器实例或 None（若配置未启用调度）。
                Scheduler instance or None (if not enabled in config).
        """
        if self._scheduler is None or rebuild:
            if self._optimizer is None:
                # 若未配置调度器，可直接返回 None
                if not getattr(self.config.training, "use_scheduler", False):
                    self.logger.info("Scheduler disabled by config.")
                    return None
                raise RuntimeError("Optimizer must be built before building scheduler.")
            self.logger.info("Building scheduler...")
            self._scheduler = build_scheduler(self.config, self._optimizer)
            self.logger.info("Scheduler built.")
        return self._scheduler


# ============================================================
# 统一入口：构建 runtime & 运行任务
# ============================================================


def create_runtime(
    config: ExperimentConfig,
    work_dir: Path | str,
    override_device: Optional[str] = None,
    logger: Optional[Any] = None,
) -> ExperimentRuntime:
    """
    @brief 从 ExperimentConfig 构建 ExperimentRuntime（不执行任何任务）。
           Build an ExperimentRuntime from ExperimentConfig (no task executed).

    @param config          实验配置对象。Experiment configuration object.
    @param work_dir        实验工作目录路径。Working directory path.
    @param override_device 可选设备字符串（如 "cpu" / "cuda:0"），优先于配置。
                           Optional device string; overrides config.training.device.
    @param logger          可选外部 logger，如果为 None 则使用项目默认 logger。
                           Optional external logger; default project logger if None.

    @return 构建好的 ExperimentRuntime，初始状态为 READY。
            Constructed ExperimentRuntime, with initial state READY.

    @note  本函数会设置随机种子与设备，但不会构建 DataLoader/模型等重资源，
           这些由 runtime 的懒加载接口在任务执行时按需构造。
           This function sets random seed and device, but does not build heavy
           resources like DataLoaders or models; those are lazily built by
           runtime methods when tasks run.
    """
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    if logger is None:
        logger = get_logger(__name__)

    # 确定设备与随机种子 Determine device and random seed
    device_str = override_device or getattr(config.training, "device", "cuda")
    device = torch.device(device_str)

    seed = getattr(config.training, "seed", 42)
    set_global_seed(seed)

    logger.info(
        f"Initialize runtime: work_dir={work_dir}, device={device}, seed={seed}"
    )

    runtime = ExperimentRuntime(
        config=config,
        work_dir=work_dir,
        device=device,
        logger=logger,
        state=RuntimeState.INIT,
    )
    # 初始化完成后切换到 READY
    runtime.transition_to(RuntimeState.READY)
    return runtime


def run_task(
    task_name: TaskName,
    runtime: ExperimentRuntime,
    **task_kwargs: Any,
) -> Any:
    """
    @brief 在给定运行时环境上执行指定名称的任务（通过注册表查找）。
           Execute a registered task by name on the given runtime.

    @param task_name  要执行的任务名称。Name of the task to execute.
    @param runtime    运行时环境对象。Experiment runtime instance.
    @param task_kwargs 传递给任务构造函数的额外参数。
                       Extra keyword arguments passed to the task constructor.

    @return 任务执行结果。Task execution result.

    @throws KeyError 若任务名称未在注册表中注册。
            Raises KeyError if the task name is not registered.

    @note  函数内部会负责状态机迁移：
           READY/COMPLETED/FAILED -> RUNNING -> COMPLETED/FAILED
           This function is responsible for state transitions:
           READY/COMPLETED/FAILED -> RUNNING -> COMPLETED/FAILED.
    """
    task_cls = TASK_REGISTRY.get(task_name)
    # 检查状态合法性 Check allowed start states
    if runtime.state not in task_cls.allowed_start_states:
        raise RuntimeError(
            f"Task '{task_name}' cannot start from runtime state "
            f"'{runtime.state.value}'. Allowed: "
            f"{[s.value for s in task_cls.allowed_start_states]}",
        )

    runtime.logger.info(f"Starting task '{task_name}' with kwargs={task_kwargs}.")

    task = task_cls(runtime, **task_kwargs)

    # 状态迁移：-> RUNNING
    prev_state = runtime.state
    runtime.transition_to(RuntimeState.RUNNING)

    try:
        task.before_run()
        result = task.run()
        task.after_run(result)
        runtime.transition_to(RuntimeState.COMPLETED)
        runtime.logger.info(f"Task '{task_name}' completed successfully.")
        return result
    except Exception as exc:
        runtime.transition_to(RuntimeState.FAILED)
        runtime.logger.error(f"Task '{task_name}' failed: {exc}", exc_info=True)
        # 是否重新抛出由你决定，此处选择重抛以便 CLI 可感知错误
        raise
    finally:
        # 如果想支持多任务串行，可以在这里决定是否自动回到 READY
        # If you want to support multiple tasks sequentially, you may choose
        # to transition back to READY here.
        if runtime.state == RuntimeState.COMPLETED:
            # 比较保守：成功后切回 READY
            runtime.transition_to(RuntimeState.READY)
