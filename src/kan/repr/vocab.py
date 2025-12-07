"""
@file vocab.py
@brief 词表 / 实体表构建与管理模块。Utilities for building & managing token / entity vocabularies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from collections import Counter

from kan.utils.logging import get_logger
from kan.data.preprocessing import PreprocessedSample

logger = get_logger(__name__)


# ============================================================
# 配置与核心数据结构 Configuration & Core Data Structures
# ============================================================


@dataclass
class VocabConfig:
    """
    @brief 词表配置，控制最小频率、大小与特殊符号等。
           Vocabulary configuration controlling min frequency, size and special tokens.
    @param min_freq 最小词频，低于该频率将映射为 UNK。Minimum frequency; tokens below go to UNK.
    @param max_size 词表最大容量（含特殊符号），None 表示不限制。Max vocab size (incl. specials), None = no limit.
    @param pad_token 填充符号 token。Padding token string.
    @param unk_token 未登录词 token。Unknown token string.
    @param bos_token 可选，序列起始 token。Optional beginning-of-sequence token.
    @param eos_token 可选，序列结束 token。Optional end-of-sequence token.
    """

    min_freq: int = 1
    max_size: Optional[int] = None
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: Optional[str] = None
    eos_token: Optional[str] = None


class Vocab:
    """
    @brief 通用词表类，同时适用于文本 token 与实体 ID。
           Generic vocabulary class for both text tokens and entity IDs.
    """

    def __init__(
        self,
        stoi: Dict[str, int],
        itos: List[str],
        pad_idx: Optional[int] = None,
        unk_idx: Optional[int] = None,
        bos_idx: Optional[int] = None,
        eos_idx: Optional[int] = None,
    ) -> None:
        """
        @brief 使用内部映射直接构造 Vocab，一般推荐通过工厂函数创建。
               Construct Vocab from internal mappings; prefer factory helpers in practice.
        @param stoi token→index 映射。Mapping from token to index.
        @param itos index→token 列表。List mapping index to token.
        @param pad_idx PAD 索引。Index of PAD token.
        @param unk_idx UNK 索引。Index of UNK token.
        @param bos_idx BOS 索引。Index of BOS token.
        @param eos_idx EOS 索引。Index of EOS token.
        """
        self.stoi = stoi
        self.itos = itos
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

    # ------------------------------------------------------------
    # 基本属性 Basic properties
    # ------------------------------------------------------------
    def __len__(self) -> int:
        """
        @brief 返回词表大小。Return vocabulary size.
        @return 词表中 token 数量。Number of tokens in vocabulary.
        """
        return len(self.itos)

    # ------------------------------------------------------------
    # 查找接口 Lookup APIs
    # ------------------------------------------------------------
    def token_to_id(self, token: str) -> int:
        """
        @brief 将 token 映射为整型 ID，如不存在则回退到 UNK。
               Map token string to integer ID, fallback to UNK if missing.
        @param token 输入 token 字符串。Input token string.
        @return 对应的整型 ID。Corresponding integer ID.
        """
        if token in self.stoi:
            return self.stoi[token]
        if self.unk_idx is not None:
            return self.unk_idx
        # 若未配置 UNK，则按需扩展词表（仅限开发调试场景）。
        # If no UNK is configured, dynamically grow the vocab (mainly for debugging).
        idx = len(self.itos)
        self.itos.append(token)
        self.stoi[token] = idx
        logger.warn(
            "Vocab.token_to_id: token %r not in vocab and no UNK configured; "
            "growing vocabulary dynamically. This is not recommended in training.",
            token,
        )
        return idx

    def id_to_token(self, idx: int) -> str:
        """
        @brief 将整型 ID 映射回 token 字符串。
               Map integer ID back to token string.
        @param idx 输入 ID。Input ID.
        @return 对应 token 字符串；越界则返回 UNK 或空串。
                Corresponding token string; UNK or empty string if out-of-range.
        """
        if 0 <= idx < len(self.itos):
            return self.itos[idx]
        if self.unk_idx is not None and 0 <= self.unk_idx < len(self.itos):
            return self.itos[self.unk_idx]
        return ""

    # ------------------------------------------------------------
    # 序列编码 / 解码 Sequence encode / decode
    # ------------------------------------------------------------
    def encode(self, tokens: Sequence[str], add_bos_eos: bool = False) -> List[int]:
        """
        @brief 将 token 序列编码为 ID 序列。
               Encode a token sequence into ID sequence.
        @param tokens 输入 token 序列。Input token sequence.
        @param add_bos_eos 是否在首尾添加 BOS/EOS。Whether to prepend/append BOS/EOS if configured.
        @return 对应的 ID 序列。Encoded ID sequence.
        """
        ids: List[int] = []
        if add_bos_eos and self.bos_idx is not None:
            ids.append(self.bos_idx)
        ids.extend(self.token_to_id(t) for t in tokens)
        if add_bos_eos and self.eos_idx is not None:
            ids.append(self.eos_idx)
        return ids

    def decode(self, ids: Sequence[int], skip_special: bool = False) -> List[str]:
        """
        @brief 将 ID 序列解码为 token 序列。
               Decode an ID sequence back to token sequence.
        @param ids 输入 ID 序列。Input ID sequence.
        @param skip_special 是否跳过特殊符号（PAD/BOS/EOS/UNK）。Whether to skip special tokens.
        @return 对应 token 序列。Decoded token sequence.
        """
        specials = {self.pad_idx, self.bos_idx, self.eos_idx, self.unk_idx}
        toks: List[str] = []
        for i in ids:
            if skip_special and i in specials:
                continue
            toks.append(self.id_to_token(i))
        return toks

    # ------------------------------------------------------------
    # 序列填充 Padding utilities
    # ------------------------------------------------------------
    def pad_batch(
        self, sequences: Sequence[Sequence[int]]
    ) -> Tuple[List[List[int]], List[int]]:
        """
        @brief 将一批变长 ID 序列填充为等长矩阵，同时返回各自真实长度。
               Pad a batch of variable-length ID sequences into a fixed-length matrix.
        @param sequences ID 序列列表。List of ID sequences.
        @return (padded, lengths)：
                - padded: 填充后的二维列表 [batch, max_len]。
                  Padded 2D list [batch, max_len].
                - lengths: 每个序列的真实长度。Real lengths of each sequence.
        """
        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths) if lengths else 0
        pad_val = self.pad_idx if self.pad_idx is not None else 0

        padded: List[List[int]] = []
        for seq in sequences:
            row = list(seq)
            if len(row) < max_len:
                row.extend([pad_val] * (max_len - len(row)))
            padded.append(row)
        return padded, lengths

    # ------------------------------------------------------------
    # 序列化 / 反序列化 Serialization helpers
    # ------------------------------------------------------------
    def to_dict(self) -> dict:
        """
        @brief 将词表导出为可 JSON 序列化的字典。
               Export vocabulary into a JSON-serializable dict.
        @return 词表状态字典。State dict of the vocabulary.
        """
        return {
            "itos": self.itos,
            "pad_idx": self.pad_idx,
            "unk_idx": self.unk_idx,
            "bos_idx": self.bos_idx,
            "eos_idx": self.eos_idx,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Vocab":
        """
        @brief 从字典重建 Vocab 实例。
               Rebuild a Vocab instance from a dict.
        @param data 先前由 to_dict 生成的字典。Dict previously produced by to_dict.
        @return 新的 Vocab 实例。New Vocab instance.
        """
        itos: List[str] = list(data["itos"])
        stoi: Dict[str, int] = {tok: i for i, tok in enumerate(itos)}
        return cls(
            stoi=stoi,
            itos=itos,
            pad_idx=data.get("pad_idx"),
            unk_idx=data.get("unk_idx"),
            bos_idx=data.get("bos_idx"),
            eos_idx=data.get("eos_idx"),
        )


# ============================================================
# 工厂函数：从语料构建词表 Factory helpers
# ============================================================


def _build_vocab_from_corpus(
    corpus: Iterable[Sequence[str]],
    cfg: VocabConfig,
) -> Vocab:
    """
    @brief 从给定语料（token 序列集合）构建 Vocab。
           Build a Vocab from a token-sequence corpus.
    @param corpus token 序列的可迭代对象。Iterable of token sequences.
    @param cfg VocabConfig 配置。Vocab configuration.
    @return 构建好的 Vocab。Constructed Vocab.
    """
    counter: Counter[str] = Counter()
    for seq in corpus:
        counter.update(seq)

    # 先按频率再按字典序排序，确保确定性。
    # Sort by frequency then lexicographically for determinism.
    items = [(tok, freq) for tok, freq in counter.items() if freq >= cfg.min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))

    # 先放入特殊符号，保持固定顺序。
    # Insert special tokens first with fixed order.
    itos: List[str] = []

    def _add_special(tok: Optional[str]) -> Optional[int]:
        if tok is None:
            return None
        if tok in itos:
            return itos.index(tok)
        itos.append(tok)
        return len(itos) - 1

    pad_idx = _add_special(cfg.pad_token)
    unk_idx = _add_special(cfg.unk_token)
    bos_idx = _add_special(cfg.bos_token)
    eos_idx = _add_special(cfg.eos_token)

    # 再插入普通 token。
    # Then append normal tokens.
    for tok, _ in items:
        if tok in itos:
            continue
        if cfg.max_size is not None and len(itos) >= cfg.max_size:
            break
        itos.append(tok)

    stoi: Dict[str, int] = {tok: i for i, tok in enumerate(itos)}
    logger.info(
        "Vocab built: size=%d (min_freq=%d, max_size=%s)",
        len(itos),
        cfg.min_freq,
        str(cfg.max_size),
    )
    return Vocab(
        stoi=stoi,
        itos=itos,
        pad_idx=pad_idx,
        unk_idx=unk_idx,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
    )


def build_text_vocab(
    samples: Iterable[PreprocessedSample],
    cfg: Optional[VocabConfig] = None,
) -> Vocab:
    """
    @brief 从预处理样本的 tokens 字段构建文本词表。
           Build text vocabulary from PreprocessedSample.tokens.
    @param samples 预处理样本可迭代对象。Iterable of PreprocessedSample.
    @param cfg VocabConfig 配置；None 使用默认值。Vocab configuration; default if None.
    @return 文本 Vocab 实例。Text vocabulary instance.
    @example
        >>> text_vocab = build_text_vocab(preprocessed_samples)
        >>> ids = text_vocab.encode(sample.tokens, add_bos_eos=True)
    """
    if cfg is None:
        cfg = VocabConfig()
    corpus = (s.tokens for s in samples)
    return _build_vocab_from_corpus(corpus, cfg)


def build_entity_vocab(
    samples: Iterable[PreprocessedSample],
    cfg: Optional[VocabConfig] = None,
) -> Vocab:
    """
    @brief 从预处理样本的实体与实体上下文构建实体 ID 词表。
           Build entity vocabulary from entities and entity contexts in samples.
    @param samples 预处理样本可迭代对象。Iterable of PreprocessedSample.
    @param cfg VocabConfig 配置；None 使用默认值，通常可关闭 BOS/EOS。Vocab configuration; None = default.
    @return 实体 Vocab 实例。Entity vocabulary instance.
    @note 实体 ID 通常已经是离散符号（如 Q76），本函数只负责做频率截断与映射。
          Entity IDs are already discrete symbols (e.g. Q76); this only does freq filtering & mapping.
    """
    if cfg is None:
        cfg = VocabConfig(bos_token=None, eos_token=None)

    def _entity_corpus():
        for s in samples:
            # 当前实体序列。
            # Current entity IDs sequence.
            if s.entities:
                yield s.entities

    return _build_vocab_from_corpus(_entity_corpus(), cfg)
