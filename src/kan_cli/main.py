"""
@file main.py
@brief KAN 命令行入口。其职责严格限定为：
       1) 解析全局与子命令参数；
       2) 加载 ExperimentConfig；
       3) 构建 ExperimentRuntime（状态机 + 运行环境）；
       4) 将控制权交给任务注册表，调度 train / evaluate / predict 等任务。
       Main entry for the KAN command-line interface, responsible for:
       1) parsing global and sub-command arguments;
       2) loading an ExperimentConfig;
       3) constructing an ExperimentRuntime (state machine + runtime context);
       4) delegating execution to tasks registered in the task registry.

@param argv 可选的参数列表（便于测试时注入）；正常运行由 sys.argv 自动提供。
       Optional list of command-line arguments (useful for testing); in normal
       usage populated automatically from sys.argv.

@note
    * main.py 不实现任何具体任务逻辑，也不直接依赖 train/evaluate/predict 的内部实现，
      而是只负责“把任务名称与参数交给 state machine + registry”。
      main.py contains no task-specific logic and does not depend on concrete
      implementations; it simply delegates task execution to the state machine
      and the registry.

    * 所有任务均应在其它模块中通过 @register_task("train") 等装饰器注册，
      以确保被 runtime 正确发现与调度。
      All concrete tasks must be registered elsewhere via decorators such as
      @register_task("train"), ensuring they are discoverable by the runtime.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List, Optional

from kan import get_logger
from kan import ExperimentConfig  # 导入仅用于类型提示 / 文档说明

from kan import load_experiment_config
from kan_cli.runtime import create_runtime, run_task

#  强制导入任务模块，确保 train/evaluate/predict 已注册到 TASK_REGISTRY
#  Force-load task module so that train/evaluate/predict are registered.
from kan_cli import tasks as _kan_cli_tasks  # noqa: F401


def _build_parser() -> argparse.ArgumentParser:
    """
    @brief 构建顶层 argparse 解析器，并注册全局选项与子命令。
           Build the top-level argparse parser with global options and sub-commands.
    @return argparse.ArgumentParser 解析器实例。
            argparse.ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="kan",
        description="KAN: Knowledge-aware Attention Network CLI",
    )

    # -------------------------------
    # 全局选项 Global options
    # -------------------------------
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help=(
            "实验配置 JSON 路径 (Experiment config JSON path, required). "
            "该文件将被解析为 ExperimentConfig，并驱动数据 / 模型 / 训练等全部流程。"
        ),
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="train",
        help=(
            "实验工作目录，用于保存模型 / 日志 / 词表等文件；默认为 ./train。"
            "Working directory for models / logs / vocabs, default './train'."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "覆盖配置中 training.device 的设备字符串，如 'cpu' 或 'cuda:0'。"
            "Override training.device in config, e.g. 'cpu' or 'cuda:0'."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="子命令 (train / evaluate / predict)。Sub-commands.",
    )

    # -------------------------------
    # train 子命令 Train sub-command
    # -------------------------------
    p_train = subparsers.add_parser(
        "train",
        help="训练 KAN 模型 (train a KAN model).",
    )
    # 训练目前完全由 ExperimentConfig 决定，无额外必需参数
    p_train.set_defaults(task_name="train")

    # -------------------------------
    # evaluate 子命令 Evaluate sub-command
    # -------------------------------
    p_eval = subparsers.add_parser(
        "evaluate",
        help=(
            "在带标签数据集上评估已有模型，输出指标与概率等。"
            "Evaluate a trained model on a labeled dataset."
        ),
    )
    p_eval.add_argument(
        "--model",
        "--checkpoint",
        dest="checkpoint",
        type=str,
        required=True,
        help=("模型 checkpoint 路径 (.pt)。" "Path to model checkpoint (.pt)."),
    )
    p_eval.add_argument(
        "--metrics",
        dest="metrics_path",
        type=str,
        default=None,
        help=(
            "评估指标输出 JSON 路径（可选；如省略则由任务内部决定默认位置）。"
            "Optional JSON path to write metrics; if omitted, task decides default."
        ),
    )
    p_eval.add_argument(
        "--probs",
        dest="probs_path",
        type=str,
        default=None,
        help=(
            "若设置，则将 (id, prob) 概率结果写入该 CSV 路径。"
            "If set, write (id, prob) probability CSV to this path."
        ),
    )
    p_eval.set_defaults(task_name="evaluate")

    # -------------------------------
    # predict 子命令 Predict sub-command
    # -------------------------------
    p_pred = subparsers.add_parser(
        "predict",
        help=(
            "在无标签测试集上进行预测，输出 id,prob。"
            "Run prediction on an unlabeled test set, outputting id,prob."
        ),
    )
    p_pred.add_argument(
        "--model",
        "--checkpoint",
        dest="checkpoint",
        type=str,
        required=True,
        help=("模型 checkpoint 路径 (.pt)。" "Path to model checkpoint (.pt)."),
    )
    p_pred.add_argument(
        "--output",
        dest="output_path",
        type=str,
        default="results.csv",
        help=(
            "预测结果 CSV 输出路径（包含 id,prob），默认 results.csv。"
            "CSV output path for prediction results (id,prob), default 'results.csv'."
        ),
    )
    p_pred.set_defaults(task_name="predict")

    return parser


def _load_config(config_path: str) -> ExperimentConfig:
    """
    @brief 从给定路径加载 ExperimentConfig。
           Load an ExperimentConfig from the given path.
    @param config_path 配置 JSON 路径。Path to the config JSON file.
    @return 解析后的 ExperimentConfig 实例。
            Parsed ExperimentConfig instance.
    @throws RuntimeError 如未找到合适的加载函数。
            Raises RuntimeError if no loader function is available.
    """
    if load_experiment_config is None:
        # 如果你使用了不同的配置加载入口，可以在这里集中修改。
        raise RuntimeError(
            "load_experiment_config is not available. "
            "Please expose a config loader as 'kan.load_experiment_config', "
            "or adjust _load_config(...) in kan_cli.main accordingly."
        )

    return load_experiment_config(Path(config_path))


def main(argv: Optional[List[str]] = None) -> None:
    """
    @brief KAN CLI 程序入口：解析命令行参数 → 加载配置 → 构建运行时 → 调度任务。
           Entry point for the KAN CLI: parse args → load config → build runtime
           → run the requested task via the registry.
    @param argv 可选的参数列表（测试时使用）；正常运行时可省略。
           Optional list of CLI arguments (useful in tests); usually omitted.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    logger = get_logger(__name__)
    logger.info("Parsed arguments: %s", args)

    # 1) 加载 ExperimentConfig
    cfg = _load_config(args.config)

    # 2) 构建 ExperimentRuntime（统一管理设备、随机种子、路径等）
    runtime = create_runtime(
        config=cfg,
        work_dir=args.work_dir,
        override_device=args.device,
        logger=get_logger("cli-runtime"),
    )

    # 3) 根据子命令收集任务参数
    task_kwargs: dict[str, Any] = {}
    command = args.command
    task_name: str = getattr(args, "task_name", command)

    if command == "evaluate":
        task_kwargs["checkpoint_path"] = args.checkpoint
        if args.metrics_path is not None:
            task_kwargs["metrics_path"] = args.metrics_path
        if args.probs_path is not None:
            task_kwargs["probs_path"] = args.probs_path
    elif command == "predict":
        task_kwargs["checkpoint_path"] = args.checkpoint
        task_kwargs["output_path"] = args.output_path
    # train 目前不需要额外参数；完全依赖 ExperimentConfig

    # 4) 调度任务：交给 runtime + 注册表处理
    run_task(
        task_name=task_name,
        runtime=runtime,
        **task_kwargs,
    )


if __name__ == "__main__":  # pragma: no cover - 交给控制台脚本使用
    main()
