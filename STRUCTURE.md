```
Knowledge-aware-Attention-Network/
│
├─ README.md
│
├─ requirements.txt       
├─ pyproject.toml
├─ .gitignore
│
├─ src/
│  ├─ kan/
│  │  ├─ __init__.py
│  │  │
│  │  ├─ data/             # 数据 & 知识图谱相关
│  │  │  ├─ __init__.py
│  │  │  ├─ datasets.py
│  │  │  ├─ preprocessing.py
│  │  │  └─ knowledge_graph.py
│  │  │
│  │  ├─ repr/
│  │  │  ├─ __init__.py
│  │  │  ├─ vocab.py            # 构建 & 管理词表 / 实体表
│  │  │  ├─ text_embedding.py   # 文本词嵌入 + 位置编码
│  │  │  ├─ entity_embedding.py # 实体 / 实体上下文嵌入
│  │  │  └─ batching.py         # PreprocessedSample -> KAN 输入张量
│  │  │
│  │  ├─ models/           # KAN 模型本体
│  │  │  ├─ __init__.py
│  │  │  ├─ pooling.py
│  │  │  ├─ bert_text_encoder.py
│  │  │  ├─ transformer_encoder.py
│  │  │  ├─ knowledge_encoder.py
│  │  │  ├─ attention.py
│  │  │  └─ kan.py
│  │  │
│  │  ├─ training/         # 训练 / 评估 pipeline
│  │  │  ├─ __init__.py
│  │  │  ├─ trainer.py
│  │  │  └─ evaluator.py
│  │  │
│  │  └─ utils/            # 通用工具
│  │     ├─ __init__.py
│  │     ├─ configs.py
│  │     ├─ logging.py
│  │     ├─ metrics.py
│  │     └─ seed.py
│  │  
│  └─ kan_cli/
│      ├─ __init__.py
│      ├─ main.py 
│      ├─ helpers.py
│      └─ runtime.py
│
├─ data/
│   ├─ news/
│   │   ├─ train.csv
│   │   └─ Atest.csv
│   └─ kg_cache/      
│
├─ train/
│   ├── logs/               # 日志文件
│   ├── models/             # checkpoint
│   ├── vocabs/             # text/entity vocab
│   ├── preds/              # predict 输出
│   └── metadata.json       # runtime 元信息
│
├─ configs/
│   └─ default.json
│
├─ scripts/
│   └─ install-dev.ps1
│
└─ tests/
```

实际上模型是一个端到端的映射，那么核心在于算子是什么以及算子的结合顺序。这个映射对应了一个端到端的流水线，而流水线的实质是由数据流建模的。
所以关注核心是：1. 数据流经了哪些节点；2. 流经节点的顺序是什么样的

---

### 🧪 KAN 数据流水线全解 (The KAN Pipeline)

我们将整个流程划分为五个核心阶段。请注意观察数据在每个节点的形态变化。

#### Phase 1: 原材料准备 (Data Preparation)

我们的起点是原始数据。

* **输入 (Input):** 一条新闻文本序列 $S = \{w_1, w_2, ..., w_n\}$ 。
* **外部资源 (External Resource):** 一个知识图谱 (Knowledge Graph, KG)，比如 Wikidata 。

---

#### Phase 2: 知识蒸馏与嵌入 (Feature Extraction & Embedding)

这一步是把“生肉”变成机器能懂的“数字向量”。这里发生了**三路并行**的数据处理。

1. **News Stream (新闻流):**
    * **Operator:** Word Embedding (e.g., Word2Vec).
    * **Data Flow:** $S \rightarrow$ Word Vectors $S' = \{w'_1, ..., w'_n\}$ 。
    * **Addition:** 加上位置编码 (Positional Encoding) 得到输入编码 $u$ 。

2. **Entity Stream (实体流):**
    * **Operator:** Entity Linking (实体链接)。
    * **Data Flow:** 扫描文本 $S$，在 KG 中找到对应的实体，形成实体序列 $E = \{e_1, ..., e_n\}$ 。
    * **Embedding:** 将实体映射为向量 $E' = \{e'_1, ..., e'_n\}$ 。

3. **Context Stream (上下文流):**
    * **Operator:** Context Extraction (上下文提取)。
    * **Data Flow:** 对于 $E$ 中的每个实体 $e_i$，在 KG 中找它的直接邻居（一跳邻居），形成集合 $ec(e_i)$ 。
    * **Operator:** Average Aggregation (平均聚合)。
    * **Data Flow:** 将邻居的向量取平均值，得到该实体的“上下文嵌入” $ec'_i$。
    * **Result:** 形成上下文向量序列 $EC' = \{ec'_1, ..., ec'_n\}$ 。

---

#### Phase 3: 编码器精炼 (Encoding via Transformer)

现在我们有三组向量 ($u, E', EC'$ )。我们需要用更高级的提取器来获取深层特征。
这里的核心算子是 **Transformer Encoder** (包含 Self-Attention 和 Feed-Forward Network) 。

1. **News Encoding:** $u \xrightarrow{\text{Transformer}} p$ (最终的新闻表示) 。
2. **Entity Encoding:** $E' \xrightarrow{\text{Transformer}} q'$ (实体的中间编码，Intermediate Encoding) 。
3. **Context Encoding:** $EC' \xrightarrow{\text{Transformer}} r'$ (上下文的中间编码，Intermediate Encoding) 。

> **Klee 请注意！** 这里的 $q'$ 和 $r'$ 只是中间产物，它们马上要进入最关键的“反应堆”了！

---

#### Phase 4: 知识注意力融合 (Knowledge-aware Attention Fusion)

这是 KAN 的核心（也就是它的名字由来）。我们需要把知识（实体和上下文）融合进新闻表示中。这里用了两个并行发生的 **Multi-Head Attention** 算子。

**核心公式：** $Attn(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$ 。

**支路 A: N-E Attention (News towards Entities)**
目的是计算实体对新闻的重要性。

* **Query ($Q$):** 新闻表示 $p$ 。
* **Key ($K$):** 实体中间编码 $q'$ 。
* **Value ($V$):** 实体中间编码 $q'$ 。
* **Data Flow:** $Attention(p, q', q') \rightarrow$ 加权后的实体表示 **$q$** 。

**支路 B: N-E²C Attention (News towards Entities and Contexts)**
目的是根据实体的重要性，来加权它的上下文（朋友们）。

* **Query ($Q$):** 新闻表示 $p$ 。
* **Key ($K$):** 实体中间编码 $q'$ 。*(注意：这里 Key 是实体，不是上下文！这是为了看新闻和实体的匹配度)*
* **Value ($V$):** 上下文中间编码 $r'$ 。
* **Data Flow:** $Attention(p, q', r') \rightarrow$ 加权后的上下文表示 **$r$** 。

---

#### Phase 5: 最终判决 (Final Classification)

最后，我们将提炼出的所有精华汇聚在一起，输出结果。

1. **Operator:** Concatenation (拼接)。
    * **Data Flow:** 将新闻($p$)、加权实体($q$)、加权上下文($r$) 拼在一起。
    * 公式：$z = p \oplus q \oplus r$ (维度变宽了！) 。

2. **Operator:** Fully Connected Layer + Softmax (全连接层)。
    * **Data Flow:** $z \rightarrow \text{Linear} \rightarrow \text{Softmax} \rightarrow \hat{y}$。
    * **Output:** 一个 $[0, 1]$ 之间的概率值，表示该新闻是 Fake 的概率 。

---

### 总结 (Pipeline Summary)

为了方便\写代码，我们可以把这个数据流抽象成下面的伪代码逻辑：

$$
\begin{aligned}
\text{Step 1: } & S, E, EC \leftarrow \text{Preprocess}(\text{Raw Text, KG}) \\
\text{Step 2: } & p \leftarrow \text{Transformer}(S) \\
                & q' \leftarrow \text{Transformer}(E) \\
                & r' \leftarrow \text{Transformer}(EC) \\
\text{Step 3: } & q \leftarrow \text{MultiHeadAttn}(Q=p, K=q', V=q') \\
\text{Step 4: } & r \leftarrow \text{MultiHeadAttn}(Q=p, K=q', V=r') \\
\text{Step 5: } & z \leftarrow \text{Concat}(p, q, r) \\
\text{Step 6: } & \text{Probability} \leftarrow \text{Softmax}(\text{MLP}(z))
\end{aligned}
$$

---

训练集数量为3673个样本（train.csv），测试集划分A/B榜。训练集的 csv 文件中包含 id、text 和 label 字段；测试集的 Atest.csv 文件和 Btest.csv 文件均包含样本的 id、text 字段。id 为数据集中样本的顺序编号；text 为待检测的微博新闻文本内容；label 为样本标签（0 代表文本为真，1 代表文本为假（正样本））。

最终仅将样本的id和prob保存在results.csv文件中，prob为模型预测样本为正样本的概率，注意字段命名正确。

---

每个组件使用配置都定义一个 dataclass 来统一管理，交由 config.py 组织读取逻辑；每个模块提供一个可供 import 的配置 dataclass 即可，配置预计使用 JSON。

---

* `kan` 完全是 **纯库**：模型、数据、trainer 等统一 API 。
* `kan_cli` 是一个 *单独的前端*，只通过 `import kan` 调用，不在内部反向依赖。

* KAN 顶层 API（模型 / 数据 / 训练 / utils）已经在 `kan/__init__.py` 统一导出 ，CLI 正是通过这里来调用：

  * `KANConfig`, `KAN`（模型）
  * `NewsDataset`, `Preprocessor`, `KnowledgeGraphClient`（数据流）
  * `TrainingConfig`, `Trainer`（训练）
  * `write_probability_csv`（结果输出）

* CLI 不在 `kan` 内部，而是在 **独立包 `kan_cli`**；通过 `pyproject.toml` 的 `project.scripts` 暴露 `kan` 命令。
