# 🌌 KAN for Weibo

> 📄 原始论文：[*KAN: Knowledge-aware Attention Network for Fake News Detection*](https://cdn.aaai.org/ojs/16080/16080-13-19574-1-2-20210518.pdf)
> 🏗 代码结构文档：见本仓库结构

---

## ✨ 项目亮点

| 能力              | 描述                                       |
| --------------- | ---------------------------------------- |
| **文本编码升级**   | BERT / Transformer Encoder 双模式          |
| **中文本土化**   | jieba / LTP 分词、bert-base-chinese              |
| **位置编码升级**  |  为实体编码器、实体-上下文编码器采用 RoPE                |
| **高解耦模块化设计**   | `kan` 为纯库；`kan_cli` 为命令行前端微内核              |
| **高解耦数据流水线** | preprocessing → vocab → batching → model |
| **动态配置系统**   | 全项目基于 dataclass + JSON 配置                |
| **整个项目由 ChatGPT 生成** | 完全 vibe coding |

KAN 不只是一个模型，而是一个**端到端知识增强的内容理解框架**。

---

# 🏛 架构总览

```
Raw Text ──► Preprocessing ──► Token Stream
                       │
                       ├──► Entity Linking ──► KG Entities
                       │
                       └──► KG Neighbor Fetch ──► Entity Contexts
```

然后三路并行：

```
Token Stream       ─► Text Encoder (BERT/Transformer)       ─► p
Entity IDs         ─► Entity Encoder (Transformer)          ─► q'
Entity Contexts    ─► Context Encoder (Transformer)         ─► r'
```

知识注意力融合：

```
q =  Attn(p, q', q')     # N-E Attention
r =  Attn(p, q', r')     # N-E²C Attention
```

最终决策：

```
z = concat(p, q, r)
ŷ = softmax(MLP(z))
```

📌 *整个过程深度融入了现代深度学习工程实践。*

---

# 📦 仓库结构（Repository Structure）

以下为 `STRUCTURE.md` 中的官方结构图（已内嵌） ：

```
Knowledge-aware-Attention-Network/
│
├─ README.md
├─ requirements.txt
├─ pyproject.toml
│
├─ src/
│  ├─ kan/              # 🧠 核心库（模型、数据、训练）
│  └─ kan_cli/          # 💻 CLI 前端
│
├─ data/
│   ├─ news/            # 训练 / 测试数据
│   └─ kg_cache/        # Wikidata 缓存
│
├─ train/
│   ├─ models/
│   └─ vocabs/
│
├─ configs/
│   └─ default.json
...
```

---

# 🚀 安装 Installation

### 1. Clone the repo

```bash
git clone https://github.com/kleedaisuki/Knowledge-aware-Attention-Network.git
cd Knowledge-aware-Attention-Network
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

或使用项目的 `pyproject.toml`：

```bash
pip install -e .
```

---

# 🧙‍♀️ 使用方法

## ✨ 训练模型

```bash
kan --config configs/experiment.json train
```

---

## ✨ 预测（Inference）

```bash
kan --config configs/experiment.json \
    predict --checkpoint train/models/model.pt \
            --output preds.csv
```

输出格式：

```
id,prob
903,0.0048
911,0.8722
...
```

*prob 为预测为“假新闻”的概率。*

---

# 🧪 数据集格式（Dataset Format）

训练集 `train.csv`：

| 字段    | 描述               |
| ----- | ---------------- |
| id    | 样本编号             |
| text  | 微博文本内容           |
| label | 0 = 真新闻, 1 = 假新闻 |

测试集 `Atest.csv`：

| 字段   | 描述   |
| ---- | ---- |
| id   | 样本编号 |
| text | 微博内容 |

---

# 🧠 模型原理（Model Overview）

### 🔹 1. 文本编码（Text Encoder）

可选：

* **BERT（推荐）** — 强语义表示
* **Transformer Encoder** — 原论文机制

### 🔹 2. 实体知识编码

* 使用实体链接工具（TagMe / Wikidata API）
* 获取实体的 **一跳邻居** 作为上下文

### 🔹 3. 双注意力融合（原论文核心创新）

| 模块              | 公式              | 作用             |
| --------------- | --------------- | -------------- |
| N-E Attention   | Attn(p, q', q') | 计算哪些实体更重要      |
| N-E²C Attention | Attn(p, q', r') | 根据实体的重要性加权其上下文 |

---

# 🧩 配置系统

所有组件均通过 dataclass 管理：

* PreprocessingConfig
* KnowledgeGraphConfig
* TextEncoderConfig
* KnowledgeEncoderConfig
* AttentionConfig
* TrainingConfig
* KANConfig
...

编辑 JSON 即可完成 **行为切换、模型结构替换、组件升级**。

---

# 🤝 致谢

原论文作者团队  
Wikidata 社区  
所有贡献者  
以及 ——  
✨ **你，阅读 README 的小可爱。** ✨

---

# 🐣 License

[GPL-3.0](./LICENSE)
