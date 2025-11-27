"""
@file main.py
@brief KAN 命令行工具主入口，负责解析子命令并分发到 train / evaluate / predict 实现。
       Main entry for the KAN command-line interface, responsible for parsing
       sub-commands and dispatching to train / evaluate / predict handlers.

@param argv 命令行参数列表，一般由解释器从系统环境中注入（例如 sys.argv）。
       argv list of command-line arguments, typically injected by the interpreter
       from the process environment (e.g., sys.argv).

@note 本模块本身不直接处理训练 / 评估 / 预测逻辑，只负责：
      1) 定义 CLI 语法（子命令与选项）；
      2) 将解析后的参数对象转发给相应的处理函数 (cli_train / cli_evaluate / cli_predict)。
      This module does not implement training / evaluation / inference logic
      itself. It only:
      1) defines CLI syntax (sub-commands and options);
      2) forwards the parsed argument namespace to the corresponding handlers
         (cli_train / cli_evaluate / cli_predict).

@example
    # 使用默认配置训练模型
    # Train a model with the default configuration
    kan train

    # 使用指定配置文件训练模型
    # Train a model with a given config JSON
    kan train --config path/to/config.json

    # 在带标签的数据集上评估已有模型，并输出指标与概率
    # Evaluate a trained model on a labeled dataset and write metrics / probs
    kan evaluate \
        --model path/to/checkpoint.pt \
        --data path/to/val.csv \
        --config path/to/config.json \
        --metrics val_metrics.json \
        --probs val_results.csv

    # 在无标签测试集上进行预测，输出 id,prob
    # Run prediction on an unlabeled test set and write id,prob
    kan predict \
        --model path/to/checkpoint.pt \
        --data path/to/test.csv \
        --config path/to/config.json \
        --output results.csv
"""

from __future__ import annotations

import argparse

from kan_cli.train import cli_train
from kan_cli.evaluate import cli_evaluate
from kan_cli.predict import cli_predict


def _build_parser() -> argparse.ArgumentParser:
    """
    @brief 构建顶层 argparse 解析器，并注册所有子命令。
           Build the top-level argparse parser and register all sub-commands.
    @return argparse.ArgumentParser 解析器实例，用于解析命令行参数。
            argparse.ArgumentParser instance used to parse CLI arguments.
    """
    parser = argparse.ArgumentParser(
        prog="kan",
        description="KAN: Knowledge-aware Attention Network CLI",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="子命令 (train / evaluate / predict)",
    )

    # ---- train ----
    p_train = subparsers.add_parser("train", help="训练 KAN 模型 (train KAN model)")
    p_train.add_argument(
        "--config",
        type=str,
        default=None,
        help="训练配置 JSON 路径；省略则使用内置 default.json。"
        "Path to training config JSON; if omitted, use built-in default.json.",
    )
    p_train.set_defaults(func=cli_train)

    # ---- evaluate ----
    p_eval = subparsers.add_parser(
        "evaluate", help="在带标签数据集上评估 KAN 模型 (evaluate KAN model)"
    )
    p_eval.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型 checkpoint 路径。Model checkpoint path.",
    )
    p_eval.add_argument(
        "--data",
        type=str,
        required=True,
        help="带标签的评估 CSV 文件路径，需包含 id,text,label。"
        "Labeled evaluation CSV path, must contain id,text,label.",
    )
    p_eval.add_argument(
        "--config",
        type=str,
        default=None,
        help="统一实验配置 JSON 路径（可选）。" "Optional experiment config JSON path.",
    )
    p_eval.add_argument(
        "--metrics",
        type=str,
        default="metrics.json",
        help="评估指标输出 JSON 路径。Output JSON path for metrics.",
    )
    p_eval.add_argument(
        "--probs",
        type=str,
        default=None,
        help="若设置，则将 (id,prob) 写入该 CSV 路径。"
        "If set, write (id,prob) to this CSV path.",
    )
    p_eval.set_defaults(func=cli_evaluate)

    # ---- predict ----
    p_pred = subparsers.add_parser(
        "predict", help="对无标签数据集进行预测，输出 id,prob。"
    )
    p_pred.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型 checkpoint 路径。Model checkpoint path.",
    )
    p_pred.add_argument(
        "--data",
        type=str,
        required=True,
        help="无标签 CSV 文件路径，需包含 id,text。"
        "Unlabeled CSV path, must contain id,text.",
    )
    p_pred.add_argument(
        "--config",
        type=str,
        default=None,
        help="实验配置 JSON 路径（可选，用于预处理 / KG 配置）。"
        "Optional experiment config JSON path (for preprocess / KG).",
    )
    p_pred.add_argument(
        "--output",
        type=str,
        default="results.csv",
        help="预测结果输出 CSV，包含 id,prob。"
        "Output CSV path for prediction results with id,prob.",
    )
    p_pred.set_defaults(func=cli_predict)

    return parser


def main() -> None:
    """
    @brief KAN CLI 程序入口：解析命令行参数并调用对应的子命令处理函数。
           Entry point for the KAN CLI: parse command-line arguments and invoke
           the corresponding sub-command handler.
    @note 命令行参数由 argparse 从进程的 sys.argv 中自动解析，本函数不显式接收参数。
          Command-line arguments are parsed from sys.argv by argparse; this
          function does not take explicit parameters.
    """
    parser = _build_parser()
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
