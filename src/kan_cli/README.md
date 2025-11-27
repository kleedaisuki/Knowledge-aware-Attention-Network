## 一、整体工作流总览

从命令行角度看，整个 CLI 的入口是：

```bash
kan <subcommand> [options]
```

子命令有：

* `kan train`
* `kan evaluate`
* `kan predict`

而 `main.py` 的职责非常单纯：

1. 用 `argparse` 定义这些子命令和参数；
2. 把解析出来的 `args` 分发给 `cli_train / cli_evaluate / cli_predict`。

也就是说：**CLI 层是一个“纯调度层”**，所有训练 / 评估 / 推理逻辑都在 `kan` 包内部 + `kan_cli` 的 glue 层里完成，`main.py` 本身是无状态的。

---

## 二、`kan train` 的工作流

从用户视角：

```bash
kan train --config path/to/config.json   # 或省略 config 用 default.json
```

从内部执行路径看，大致是这几步：

### 1. 解析配置路径

* 如果没有 `--config`：
  用 `importlib.resources.files("kan.configs") / "default.json"` 去拿默认配置路径。
* 有 `--config`：
  就用用户给的。

这一步是 **完全 decoupled** 的：
CLI 不关心 JSON 内容，只关心“路径”。

### 2. 加载实验配置

* 调用 `kan.load_experiment_config(config_path)`（你在 `kan.utils.config` 里实现，并在 `kan.__init__` 中 re-export 过）。
* 得到一个 `exp_cfg`，它可以是：

  * dict，或者
  * dataclass/namespace（CLI 做了兼容）。

### 3. 从 `exp_cfg` 中抽取子配置

通过 `_get_subconfig(exp_cfg, name, cls)` 这种模式，构造出：

* `DatasetConfig`（CSV 路径 / batch_size / 是否 shuffle 等）
* `PreprocessConfig`（lowercase / 去 URL / max_tokens / enable_kg 等）
* `KnowledgeGraphConfig`（Wikidata SPARQL endpoint / timeout / cache_dir 等）
* `TrainingConfig`（epochs / lr / warmup / grad_clip / device / seed / output_dir）
* `KANConfig`（text encoder / knowledge encoder / attention / num_classes / dropout 等）

解耦点在这儿：**CLI 不写死任何超参，只把 JSON 解析成 dataclass**，模型超参仍然完全由 `kan` 管。

### 4. 构建数据流组件

根据这些配置，构造：

* `NewsDataset(dataset_cfg)`：

  * 读取 CSV，构造 `List[NewsSample]`，并提供 `batch_iter()`（yield `List[NewsSample]`）
* `KnowledgeGraphClient(kg_cfg)`（如果启用 KG）
* `Preprocessor(preprocess_cfg, kg_client)`：

  * 文本清洗 + 简单 token 化
  * KG 实体链接 + 一跳邻居上下文（enable_kg=True && 有 kg_client 时）

这一层的职责是：

> **Raw CSV → (tokens, entities, entity_contexts)**

而不是直接返回张量。张量构造放到更后面的 batching 层。

### 5. 构建 KAN 模型

* 用 `KANConfig` 初始化 `KAN`：

  * 内部构造 Transformer 文本编码器、知识编码器、N-E / N-E²C 注意力、分类头等，并做 d_model 一致性检查。

CLI 做的一件事是：

* 从 `KANConfig.text.encoder.d_model` 里拿出 `embed_dim`，用于我们在 CLI 这边构造哈希 embedding 的向量维度，保证和模型内部一致。

### 6. 包一层 `KANForTrainer`（关键）

因为你的 `Trainer` 是这样设计的：

* 它期望 `model(**inputs)` 返回 **一个 Tensor**，后面会：

  ```python
  logits_flat = logits.view(-1)
  labels_float = labels.float().view(-1)
  loss = BCEWithLogitsLoss(logits_flat, labels_float)
  ```

* 并且 `Trainer` 会把 `labels` 当作 shape `(B,)` 的张量来看。

而 `KAN.forward` 的签名是：

```python
logits, aux = model(
    news_embeddings,
    news_padding_mask,
    entity_embeddings,
    entity_context_embeddings,
    entity_padding_mask,
    return_attn_weights=False,
)
```

所以：

* 为了让 `Trainer` 不被破坏，我们在 CLI 写了 `KANForTrainer`，包装内部的 `KAN`，对外暴露 `forward(**inputs) -> logits`。

✅ 这个“**适配器模式**”是很对味的：
训练循环完全由 `Trainer` 管，模型结构完全由 `KAN` 管，CLI 只负责两者之间的接口适配。

> 但这里有一个 **重要冲突**，后面单独说。

### 7. 构造训练 batch（`iter_batches_for_training`）

这部分逻辑在 `kan_cli.batching` 里：

**数据流是：**

1. `NewsDataset.batch_iter()` 给你 `List[NewsSample]`。

2. `Preprocessor.preprocess_batch` 把它变成 `List[PreprocessedSample]`：包含 `tokens / entities / entity_contexts / label`。

3. `build_batch_tensors(pre_samples, embed_dim, with_labels=True)`：

   * 用 `StringHashEmbedding(dim)`（哈希嵌入）把字符串变成向量：

     * 对 tokens → `news_embeddings`
     * 对 entity IDs → `entity_embeddings`
     * 对 entity contexts → 先平均邻居 → `entity_context_embeddings`
   * 同时构建：

     * `news_padding_mask`（True=padding）
     * `entity_padding_mask`
     * labels → `labels: LongTensor`
   * 返回一个 dict，键包括：

     * `"news_embeddings"`, `"news_padding_mask"`,
     * `"entity_embeddings"`, `"entity_context_embeddings"`, `"entity_padding_mask"`,
     * `"labels"`。

4. `iter_batches_for_training(...)` 就 yield 这样的 dict，完全对标 `Trainer` 的需求：

   * `labels` 单独存在；
   * 其它张量都作为模型输入。

### 8. 把 batch 迭代器交给 `Trainer`

* 在 CLI 中创建：

  ```python
  train_data = iter_batches_for_training(dataset, preprocessor, embed_dim)
  trainer = Trainer(model=wrapped_model, train_data=train_data, cfg=training_cfg)
  trainer.train()
  ```

* `Trainer.train()` 会：

  * `for epoch in range(num_epochs):`
  * `for batch in self.train_data:`

    * `_train_one_batch(batch)` → 前向、反向、更新
  * 每个 epoch 保存 checkpoint（按 `save_every`）。

这里等于实现了：

> CLI 只负责把“数据批次 mapping”按 `Trainer` 期望的格式吐出来，训练状态（optimizer / scheduler / 梯度裁剪 / checkpoint）都由 `Trainer` 管。

---

## 三、`kan evaluate` 的工作流

从用户角度：

```bash
kan evaluate \
  --model path/to/epochX.pt \
  --data path/to/val.csv \
  --config path/to/exp.json \
  --metrics val_metrics.json \
  --probs val_results.csv
```

内部流程和 train 很像，只是不用 `Trainer`，而是：

1. 同样用 `DatasetConfig + PreprocessConfig + KGConfig + KANConfig` 构造 dataset & preprocessor & model。
2. 用 `iter_batches_for_inference(..., with_labels=True)` 产出 batch：

   * 和训练的 batch 格式一致，但额外附一个 `ids` 列表用于结果对齐。
3. 走手写推理循环：

   * `wrapped_model(**batch_inputs)` → logits
   * 取正类 logit（如果是 2 类的话是 `logits[..., 1]`），做 sigmoid 得到 prob
   * 累积 `labels / probs / ids`。
4. 调 `kan.compute_binary_classification_metrics` 计算精度、F1、AUC 等。
5. 把 metrics 写 JSON；如果指定了 `--probs`，就也输出 `id,prob` CSV。

**特点：**

* 评估不依赖 `Trainer`，而是完全自己控制 forward loop；
* 所有计算指标的逻辑都在 `kan.utils.metrics` 中实现，CLI 只是调用。

逻辑上是 **干净且解耦的**。

---

## 四、`kan predict` 的工作流

从用户角度：

```bash
kan predict \
  --model path/to/epochX.pt \
  --data data/news/Atest.csv \
  --config path/to/exp.json \
  --output results.csv
```

内部流程基本是 `evaluate` 的子集：

1. 用 `DatasetConfig` 构造 dataset，但强制不 shuffle，且不要求 `label` 列（无标签数据）。
2. `Preprocessor` + `iter_batches_for_inference(..., with_labels=False)`：

   * batch 不包含 labels，只包含：

     * `ids`, `news_embeddings`, `news_padding_mask`, 以及实体相关张量。
3. `wrapped_model(**batch_inputs)` → logits → 正类 prob → 累积 `ids` & `probs`。
4. 最后用 `write_probability_csv(id, prob)` 输出 `results.csv`（或者 fallback 自己写）。

这里也完全不依赖 `Trainer`。
