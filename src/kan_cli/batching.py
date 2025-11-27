"""
@file batching.py
@brief 将预处理后的样本批次转换为 KAN 所需的张量输入。
       Convert preprocessed samples batches into tensor inputs required by KAN.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Mapping, Sequence

import torch

import kan  # 使用 NewsSample / PreprocessedSample / Preprocessor / NewsDataset 等 :contentReference[oaicite:4]{index=4}
from kan_cli.embedding import StringHashEmbedding


def _ensure_tokens(tokens: Sequence[str]) -> List[str]:
    """
    @brief 保证每条样本至少有一个 token。
           Ensure that each sample has at least one token.
    """
    if not tokens:
        return ["<empty>"]
    return list(tokens)


class TrainingDataIterable:
    """
    @brief 训练数据可重启 iterable，每次迭代都会重新遍历底层数据集并生成 batch。
           Restartable iterable over training data; each iteration runs a fresh
           pass over the underlying dataset and yields batches.
    """

    def __init__(
        self,
        dataset: "kan.NewsDataset",  # type: ignore[name-defined]
        preprocessor: "kan.Preprocessor",  # type: ignore[name-defined]
        embed_dim: int,
        embedder: StringHashEmbedding,
    ) -> None:
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.embed_dim = embed_dim
        self.embedder = embedder

    def __iter__(self):
        """
        @brief 返回一个新的 batch 生成器，用于本轮训练 epoch。
               Return a fresh batch generator for the current training epoch.
        """
        return iter_batches_for_training(
            dataset=self.dataset,
            preprocessor=self.preprocessor,
            embed_dim=self.embed_dim,
            embedder=self.embedder,
        )


def build_batch_tensors(
    samples: Sequence["kan.PreprocessedSample"],  # type: ignore[name-defined]
    embed_dim: int,
    with_labels: bool,
    embedder: StringHashEmbedding,
) -> Dict[str, Any]:
    """
    @brief 将一批 PreprocessedSample 转换为 KAN.forward 所需张量。
           Convert a batch of PreprocessedSample into tensors for KAN.forward.
    @param samples 预处理后的样本列表。List of preprocessed samples.
    @param embed_dim 嵌入维度，应与 KANConfig 的 d_model 一致。
           Embedding dimension, should match KANConfig's d_model.
    @param with_labels 是否包含标签（训练 / 评估）。Whether labels are present.
    @return 包含 news_embeddings, entity_embeddings 等键的字典。
            Mapping containing news_embeddings, entity_embeddings, etc.
    """
    if len(samples) == 0:
        raise ValueError("build_batch_tensors: empty sample batch is not allowed.")

    B = len(samples)
    D = embed_dim

    # -------- 文本部分 --------
    token_lists: List[List[str]] = [_ensure_tokens(s.tokens) for s in samples]
    max_len_text = max(len(toks) for toks in token_lists)

    news_embeddings = torch.zeros(B, max_len_text, D, dtype=torch.float32)
    news_padding_mask = torch.ones(B, max_len_text, dtype=torch.bool)

    for i, toks in enumerate(token_lists):
        for j, tok in enumerate(toks):
            if j >= max_len_text:
                break
            news_embeddings[i, j] = embedder(tok)
            news_padding_mask[i, j] = False  # False 表示非 padding

    batch: Dict[str, Any] = {
        "news_embeddings": news_embeddings,
        "news_padding_mask": news_padding_mask,
    }

    # -------- 实体部分 --------
    entity_lists: List[Sequence[str]] = [s.entities for s in samples]
    max_len_ent = max((len(ents) for ents in entity_lists), default=0)

    if max_len_ent > 0:
        entity_embeddings = torch.zeros(B, max_len_ent, D, dtype=torch.float32)
        entity_ctx_embeddings = torch.zeros(B, max_len_ent, D, dtype=torch.float32)
        entity_padding_mask = torch.ones(B, max_len_ent, dtype=torch.bool)

        for i, s in enumerate(samples):
            ents = s.entities
            ctxs = s.entity_contexts
            for j, eid in enumerate(ents):
                if j >= max_len_ent:
                    break
                entity_embeddings[i, j] = embedder(eid)
                # 上下文向量：平均所有邻居；为空则保持零向量
                if j < len(ctxs) and len(ctxs[j]) > 0:
                    ctx_vecs = [embedder(c) for c in ctxs[j]]
                    ctx_stack = torch.stack(ctx_vecs, dim=0)
                    entity_ctx_embeddings[i, j] = ctx_stack.mean(dim=0)
                entity_padding_mask[i, j] = False

        batch["entity_embeddings"] = entity_embeddings
        batch["entity_context_embeddings"] = entity_ctx_embeddings
        batch["entity_padding_mask"] = entity_padding_mask

    # -------- 标签部分 --------
    if with_labels:
        labels: List[int] = []
        for s in samples:
            if s.label is None:
                raise RuntimeError(
                    "build_batch_tensors(with_labels=True) but sample.label is None."
                )
            labels.append(int(s.label))
        batch["labels"] = torch.tensor(labels, dtype=torch.long)

    return batch


def iter_batches_for_training(
    dataset: "kan.NewsDataset",  # type: ignore[name-defined]
    preprocessor: "kan.Preprocessor",  # type: ignore[name-defined]
    embed_dim: int,
    embedder: StringHashEmbedding,
) -> Iterator[Mapping[str, Any]]:
    """
    @brief 基于 NewsDataset 和 Preprocessor 生成训练用 batch（字典形式）。
           Generate training batches as mappings, using NewsDataset and Preprocessor.
    @param dataset 新闻数据集。News dataset.
    @param preprocessor 预处理器。Preprocessor instance.
    @param embed_dim 嵌入维度。Embedding dimension.
    @return 可迭代的 batch，每个 batch 符合 Trainer 要求。
            Iterable of batches; each batch is a mapping for Trainer.
    """
    for raw_batch in dataset.batch_iter():
        pre_samples = preprocessor.preprocess_batch(raw_batch)
        batch = build_batch_tensors(
            pre_samples,
            embed_dim=embed_dim,
            with_labels=True,
            embedder=embedder,
        )
        yield batch


def iter_batches_for_inference(
    dataset: "kan.NewsDataset",  # type: ignore[name-defined]
    preprocessor: "kan.Preprocessor",  # type: ignore[name-defined]
    embed_dim: int,
    embedder: StringHashEmbedding,
    with_labels: bool,
) -> Iterator[Dict[str, Any]]:
    """
    @brief 为评估/预测生成 batch，支持有无标签两种模式。
           Generate batches for evaluation/prediction, with or without labels.
    @param dataset 新闻数据集。News dataset.
    @param preprocessor 预处理器。Preprocessor instance.
    @param embed_dim 嵌入维度。Embedding dimension.
    @param with_labels 是否包含标签。Whether labels are included.
    @return 可迭代 batch 字典。Iterable of batch mappings.
    """
    for raw_batch in dataset.batch_iter():
        pre_samples = preprocessor.preprocess_batch(raw_batch)
        batch = build_batch_tensors(
            pre_samples,
            embed_dim=embed_dim,
            with_labels=with_labels,
            embedder=embedder,
        )
        ids = [s.id for s in pre_samples]
        batch["ids"] = ids
        yield batch
