# KAN CLI 使用手册

**Knowledge-aware Attention Network — Command-Line Interface Guide**

---

## 目录

* [简介](#简介)
* [CLI 整体结构](#cli-整体结构)
* [快速开始](#快速开始)
* [命令说明](#命令说明)

  * [1. `train`](#1-train)
  * [2. `evaluate`](#2-evaluate)
  * [3. `predict`](#3-predict)
* [任务注册机制](#任务注册机制)
* [工作目录结构](#工作目录结构)
* [常见问题](#常见问题)

---

## 简介

KAN CLI 是运行本项目的**统一入口**，用于：

* 🚀 **训练模型**（`train`）
* 📊 **在带标签集上评估模型**（`evaluate`）
* 🔮 **在无标签数据上生成预测**（`predict`）

CLI 的核心设计原则：

* **所有行为由配置文件 ExperimentConfig 定义**
* **运行流程由 ExperimentRuntime 构建（状态机 + 环境）**
* **任务通过 task registry 注册、统一调度**
* **CLI 本身只做参数解析与调度，不包含业务逻辑**

---

## CLI 整体结构

```
kan --config <config.json> [--work-dir DIR] [--device DEV] <command> [options...]
```

组件含义：

| 参数/组件        | 作用                                  |
| ------------ | ----------------------------------- |
| `--config`   | 指定实验配置文件（必填）                        |
| `--work-dir` | 指定模型/词表/日志的输出根目录（默认：`train/`）       |
| `--device`   | 覆盖配置中的 device（如：`cpu`、`cuda:0`）     |
| `<command>`  | 任务：`train` / `evaluate` / `predict` |
| `[options]`  | 任务特定的参数                             |

---

## 快速开始

### 1. 准备配置文件

假设你已经写好了一个完整的：

```
configs/experiment.json
```

### 2. 开始训练

```bash
kan --config configs/experiment.json train
```

输出会保存在：

```
train/
  ├── models/
  ├── logs/
  ├── vocabs/
  └── ...
```

### 3. 评估模型

```bash
kan --config configs/experiment.json \
    evaluate --checkpoint train/models/model.pt \
             --metrics eval.json \
             --probs eval_probs.csv
```

### 4. 预测（无标签集）

```bash
kan --config configs/experiment.json \
    predict --checkpoint train/models/model.pt \
            --output preds.csv
```

---

## 命令说明

## 1. `train`

训练 KAN 模型（使用配置中的 dataset/train text encoder/knowledge encoder 等设置）。

```
kan --config <config.json> train
```

没有任务特定参数，因为训练完全由配置文件决定。

---

## 2. `evaluate`

在**带标签数据集**上评估模型，生成 metrics JSON 与概率 CSV。

```
kan --config <config.json> \
    evaluate \
      --checkpoint <model.pt> \
      [--metrics METRICS.json] \
      [--probs PROBS.csv]
```

参数：

| 参数             | 说明                    |
| -------------- | --------------------- |
| `--checkpoint` | 模型参数文件（必须）            |
| `--metrics`    | 输出指标 JSON（可选）         |
| `--probs`      | 输出 (id, prob) CSV（可选） |

---

## 3. `predict`

在**无标签数据集**上预测，输出 (id, prob)。

```
kan --config <config.json> \
    predict \
      --checkpoint <model.pt> \
      [--output OUT.csv]
```

参数：

| 参数             | 说明                        |
| -------------- | ------------------------- |
| `--checkpoint` | 模型参数文件（必须）                |
| `--output`     | 输出预测 CSV，默认：`results.csv` |

---

## 任务注册机制

KAN CLI 使用一个简洁优雅的**任务注册表**来调度任务：

```python
@register_task("train")
class TrainTask(TaskBase):
    ...
```

所有任务类必须：

* 通过 `@register_task("name")` 注册
* 覆盖 `run()` 方法
* 在 `allowed_start_states` 中声明可运行状态

CLI 调度流程：

```
main.py → parse args → create_runtime → run_task(task_name)
```

---

## 工作目录结构

默认 `work-dir = train/`：

```
train/
  ├── logs/               # 日志文件
  ├── models/             # checkpoint
  ├── vocabs/             # text/entity vocab
  ├── preds/              # predict 输出
  └── metadata.json       # runtime 元信息
```

你也可以通过：

```
--work-dir my_experiment/
```

来自定义。

---

## 常见问题

### 1. 为什么运行时报 “task not found”？

因为你的任务没有被 import，从而没有注册到 `TASK_REGISTRY`。

解决方式：

```python
from kan_cli import tasks
```

CLI 里已经自动强制 import 过一次。

---

### 2. evaluate/predict 数据从哪里来的？

当前版本中，evaluation/prediction 共享配置中的 dataset（例如 val/test）。

之后可扩展为显式 `dataset.val_path` / `dataset.test_path`。

---

### 3. 我能扩展新任务吗？

可以！只需：

```python
@register_task("my_task")
class MyTask(TaskBase):
    allowed_start_states = {...}

    def run(self):
        ...
```

然后：

```
kan --config xxx.json my_task
``
