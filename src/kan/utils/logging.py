"""
@file logging.py
@brief 项目统一日志工具，提供微秒精度时间戳和 stdout/stderr 分流。
       Unified logging utilities with microsecond timestamps and stdout/stderr routing.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


class _ExactLevelFilter(logging.Filter):
    """
    @brief 只允许指定等级的日志记录通过的过滤器。Filter that only allows records of an exact level.
    @param level 需要通过的日志等级（例如 logging.INFO）。Log level to pass (e.g. logging.INFO).
    @note 主要用于将 INFO 日志单独分流到 stdout。Mainly used to route INFO logs to stdout only.
    """

    def __init__(self, level: int) -> None:
        self.level = level
        super().__init__()

    def filter(self, record: logging.LogRecord) -> bool:
        """
        @brief 判断当前日志记录是否通过过滤。Decide whether the record passes the filter.
        @param record 日志记录对象。Logging record instance.
        @return True 表示通过；False 表示丢弃。True to keep the record, False to drop it.
        """
        return record.levelno == self.level


def _create_formatter() -> logging.Formatter:
    """
    @brief 创建带微秒时间戳且无空格的日志格式化器。Create formatter with microsecond timestamps and no spaces.
    @return logging.Formatter 实例。A logging.Formatter instance.
    @note 时间戳格式为: YYYY-MM-DD-HH:MM:SS.microseconds
          Timestamp format: YYYY-MM-DD-HH:MM:SS.microseconds
    """
    fmt = "%(asctime)s-%(levelname)s-%(name)s-%(message)s"
    # Python 3.11 支持 %f 微秒；这里用 '-' 连接日期和时间，且整体无空格
    # Python 3.11 supports %f for microseconds; we use '-' between date and time and no spaces overall
    datefmt = "%Y-%m-%d-%H:%M:%S.%f"
    return logging.Formatter(fmt=fmt, datefmt=datefmt)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    @brief 获取带标准配置的 logger，INFO 打到 stdout，WARN/ERROR 打到 stderr。
           Get a logger with standard config: INFO to stdout, WARN/ERROR to stderr.
    @param name 日志名称（通常为 __name__ 或 "kan"）。Logger name (e.g. __name__ or "kan").
    @return 已配置好的 logger 实例。A configured logger instance.
    @note 该函数具有幂等性，多次调用不会重复添加 handler。
          This function is idempotent; multiple calls will not add duplicate handlers.
    """
    logger_name = name if name is not None else "kan"
    logger = logging.getLogger(logger_name)

    # 避免重复配置：如果已经标记过，就直接返回
    # Avoid double configuration: if already marked, just return
    if getattr(logger, "_kan_configured", False):
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = _create_formatter()

    # --- INFO -> stdout ---
    info_handler = logging.StreamHandler(stream=sys.stdout)
    info_handler.setLevel(logging.INFO)
    info_handler.addFilter(_ExactLevelFilter(logging.INFO))
    info_handler.setFormatter(formatter)

    # --- WARNING/ERROR -> stderr ---
    err_handler = logging.StreamHandler(stream=sys.stderr)
    err_handler.setLevel(logging.WARNING)  # 包含 WARNING、ERROR、CRITICAL
    err_handler.setFormatter(formatter)

    logger.addHandler(info_handler)
    logger.addHandler(err_handler)

    # 打个内部标记，避免重复初始化
    # Mark as configured to avoid re-initialization
    setattr(logger, "_kan_configured", True)

    return logger


# 提供一个模块级默认 logger，方便 utils 内部直接用
# Provide a module-level default logger for convenience
_logger = get_logger("kan")


def info(msg: str, *args, **kwargs) -> None:
    """
    @brief 记录 info 级别日志，输出到 stdout。Log an info-level message to stdout.
    @param msg 日志消息模板。Log message template.
    @param args 格式化参数（与 logging.info 一致）。Positional formatting args.
    @param kwargs 关键字参数（与 logging.info 一致）。Keyword arguments for logging.
    """
    _logger.info(msg, *args, **kwargs)


def warn(msg: str, *args, **kwargs) -> None:
    """
    @brief 记录 warn 级别日志，输出到 stderr。Log a warning-level message to stderr.
    @param msg 日志消息模板。Log message template.
    @param args 格式化参数。Positional formatting args.
    @param kwargs 关键字参数。Keyword arguments for logging.
    @note 内部调用的是 logging.WARNING 等级。Internally uses logging.WARNING level.
    """
    _logger.warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs) -> None:
    """
    @brief 记录 error 级别日志，输出到 stderr。Log an error-level message to stderr.
    @param msg 日志消息模板。Log message template.
    @param args 格式化参数。Positional formatting args.
    @param kwargs 关键字参数。Keyword arguments for logging.
    """
    _logger.error(msg, *args, **kwargs)
