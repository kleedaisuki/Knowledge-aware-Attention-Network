"""
@file embedding.py
@brief 基于字符串哈希的确定性向量映射，用于将 tokens / 实体 ID 映射到固定维度向量。
       Deterministic string-to-vector mapping based on hashing, used to map
       tokens / entity IDs into fixed-dimensional vectors.
"""

from __future__ import annotations

import hashlib
from typing import Dict

import torch
from torch import Tensor


class StringHashEmbedding:
    """
    @brief 将任意字符串映射到 R^d 的确定性向量，不依赖训练权重。
           Deterministically map arbitrary strings into R^d without trainable weights.
    """

    def __init__(self, dim: int) -> None:
        """
        @brief 构造函数，指定向量维度。
               Constructor specifying embedding dimension.
        @param dim 嵌入维度。Embedding dimension.
        """
        self.dim = dim
        self._cache: Dict[str, Tensor] = {}

    def _hash_bytes(self, key: str, n_bytes: int) -> bytes:
        """
        @brief 使用 SHA256 迭代生成足够的字节数。
               Use SHA256 to generate enough bytes iteratively.
        @param key 输入字符串。Input string.
        @param n_bytes 需要的字节数。Number of bytes required.
        @return 字节序列。Byte sequence.
        """
        base = key.encode("utf-8")
        data = b""
        counter = 0
        while len(data) < n_bytes:
            h = hashlib.sha256(base + counter.to_bytes(4, "little")).digest()
            data += h
            counter += 1
        return data[:n_bytes]

    def __call__(self, key: str) -> Tensor:
        """
        @brief 获取字符串对应的嵌入向量。
               Get embedding vector corresponding to the given string.
        @param key 输入字符串。Input string.
        @return 形状为 (dim,) 的浮点向量。Float vector of shape (dim,).
        """
        if key in self._cache:
            return self._cache[key]

        n_bytes = self.dim * 4
        raw = self._hash_bytes(key, n_bytes)
        ints = torch.frombuffer(raw, dtype=torch.uint8).view(-1, 4).to(torch.int32)
        # 合并成 32-bit 整数
        vals = (
            (ints[:, 0].to(torch.int64))
            | (ints[:, 1].to(torch.int64) << 8)
            | (ints[:, 2].to(torch.int64) << 16)
            | (ints[:, 3].to(torch.int64) << 24)
        )
        vals = vals.to(torch.float32)
        # 映射到 [-1, 1]
        vec = (vals / (2**31) - 1.0).view(-1)
        if vec.numel() != self.dim:
            vec = vec[: self.dim]
        self._cache[key] = vec
        return vec
