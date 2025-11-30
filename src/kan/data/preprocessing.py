"""
@file preprocessing.py
@brief 文本与知识图谱联合预处理模块，将原始样本映射为 (S, E, EC) 形式。
       Text & knowledge-graph joint preprocessing module, mapping raw samples to (S, E, EC).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Any, Sequence, Iterable, Dict

from kan.utils.logging import get_logger
from kan.data.datasets import NewsSample
from kan.data.knowledge_graph import KnowledgeGraphClient
from ltp import LTP

logger = get_logger(__name__)


# ============================================================
# 配置与数据结构 Configuration & Data Structures
# ============================================================


@dataclass
class PreprocessConfig:
    """
    @brief 预处理阶段的配置参数。Configuration for preprocessing stage.
    @param do_lowercase 是否将文本转换为小写。Whether to lowercase text.
    @param remove_urls 是否移除 URL。Whether to strip URLs.
    @param remove_user_mentions 是否移除用户提及（例如 @username）。Whether to strip user mentions.
    @param max_tokens 最大 token 数，超过则截断（0 表示不限制）。Max number of tokens (0 = no limit).
    @param enable_kg 是否启用知识图谱相关处理。Whether to enable KG-based processing.
    @param tokenizer 分词器类型，例如 "ltp" 或 "simple"。Type of tokenizer, e.g. "ltp" or "simple".
    """

    do_lowercase: bool = True
    remove_urls: bool = True
    remove_user_mentions: bool = True
    max_tokens: int = 256
    enable_kg: bool = True
    tokenizer: str = "ltp"
    ltp_model_kwargs: Optional[Dict] = None


@dataclass
class PreprocessedSample:
    """
    @brief 预处理后的单条样本结构。One sample after preprocessing.
    @param id 样本 ID。Sample ID.
    @param tokens 文本 token 序列 S。Token sequence S from text.
    @param entities 实体 ID 序列 E。Entity ID sequence E.
    @param entity_contexts 每个实体的一跳邻居列表 EC。Entity contexts EC per entity.
    @param label 标签（如果有）。Label if available.
    """

    id: int
    tokens: List[str]
    entities: List[str]
    entity_contexts: List[List[str]]
    label: Optional[int] = None


# ============================================================
# 预处理核心类 Preprocessor
# ============================================================


class Preprocessor:
    """
    @brief 负责将 NewsSample 转换为 (tokens, entities, entity_contexts) 的预处理器。
           Preprocessor that converts NewsSample into (tokens, entities, entity_contexts).
    """

    def __init__(
        self, cfg: PreprocessConfig, kg_client: KnowledgeGraphClient = None
    ) -> None:
        """
        @brief 初始化预处理器。Initialize preprocessor.
        @param cfg PreprocessConfig 配置对象。Preprocess configuration object.
        @param kg_client 知识图谱客户端，需提供 link_entities / get_entity_contexts 接口。
               Knowledge graph client providing link_entities / get_entity_contexts.
        @note kg_client 预期接口：
              - link_entities(text: str) -> Sequence[str]
              - get_entity_contexts(entities: Sequence[str]) -> Sequence[Sequence[str]]
        """
        self.cfg = cfg
        self.kg_client = kg_client

        # 决定分词器类型（默认使用 LTP，如不可用则回退到 simple）
        tokenizer = getattr(self.cfg, "tokenizer", "ltp")
        self._tokenizer_type: str
        self._ltp: Any = None

        if tokenizer == "ltp":
            # 允许通过 config.ltp_model_kwargs 追加更多底层模型参数
            ltp_kwargs = getattr(self.cfg, "ltp_model_kwargs", None)

            try:
                self._ltp = LTP(**(ltp_kwargs if isinstance(ltp_kwargs, dict) else {}))
                self._tokenizer_type = "ltp"

                def _try_get(kwargs: dict, key: Any, default: Any = None):
                    value = kwargs.get(key)
                    return value if value is not None else default

                logger.info(
                    "Preprocessor: initialized LTP tokenizer "
                    "(model=%s, cache_dir=%s, local_files_only=%s).",
                    _try_get(ltp_kwargs, "pretrained_model_name_or_path", "LTP/small"),
                    _try_get(ltp_kwargs, "cache_dir", None),
                    _try_get(ltp_kwargs, "local_files_only", False),
                )
            except Exception as exc:  # pragma: no cover - 防御性降级
                logger.error(
                    "Preprocessor: failed to initialize LTP with kwargs %s, "
                    "falling back to simple regex tokenizer. Error: %s",
                    ltp_kwargs,
                    exc,
                )
                self._ltp = None
                self._tokenizer_type = "simple"
        else:
            self._tokenizer_type = "simple"
            logger.info(
                "Preprocessor: using simple regex-based tokenizer (tokenizer=%s).",
                tokenizer,
            )

        if self.cfg.enable_kg and self.kg_client is None:
            logger.warning(
                "Preprocessor: enable_kg=True 但未提供 kg_client，将只做文本预处理。"
                "Preprocessor: enable_kg=True but no kg_client given, fallback to text-only."
            )

    # ------------------------------------------------------------
    # 文本清洗与 token 化 Text cleaning & tokenization
    # ------------------------------------------------------------
    def _clean_text(self, text: str) -> str:
        """
        @brief 按配置清洗原始文本（去 URL / 提及 / 空白等）。
               Clean raw text according to config (URLs, mentions, whitespaces).
        @param text 原始文本。Raw text.
        @return 清洗后的文本。Cleaned text.
        """
        orig = text

        if self.cfg.do_lowercase:
            text = text.lower()

        if self.cfg.remove_urls:
            # 非严格 URL 匹配，仅用于社交文本清洗
            text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        if self.cfg.remove_user_mentions:
            # 类微博/推特的 @username
            text = re.sub(r"@\w+", " ", text)

        # 归一化空白
        text = re.sub(r"\s+", " ", text).strip()

        logger.debug(f"Clean text from: [{orig[:30]}...] to [{text[:30]}...]")
        return text

    def _tokenize(self, text: str) -> List[str]:
        """
        @brief 文本 token 化：优先使用 LTP 中文分词器，其次回退到正则 tokenizer。
               Text tokenization: prefer LTP-based Chinese word segmentation, fallback to regex tokenizer.
        @param text 清洗后的文本。Cleaned text.
        @return token 列表。List of tokens.
        @note
            - 当 cfg.tokenizer="ltp" 且环境中安装了 LTP 时，使用 LTP 完成分词；
              否则退回到简单的正则分词实现。
            - 该接口保持不变，上层只感知 token 序列长度与内容的变化。
              The API is kept stable; only the actual token sequence may change.
        """
        tokens: List[str]

        # 只在配置启用且 _ltp 实例可用时才尝试 LTP
        if self._tokenizer_type == "ltp" and self._ltp is not None:
            try:
                ltp = self._ltp

                # --- 情况 1：老版 LTP（有 seg 方法） ---
                if hasattr(ltp, "seg"):
                    # 兼容老接口：seg(...) 既可能直接返回 segments，也可能返回 (segments, hidden)
                    seg_result = ltp.seg([text])
                    if isinstance(seg_result, tuple):
                        segments = seg_result[0]
                    else:
                        segments = seg_result
                    # 预期为 List[List[str]]，取第一句
                    tokens = list(segments[0])

                # --- 情况 2：新版 LTP（Transformers 版，使用 pipeline 接口） ---
                elif hasattr(ltp, "pipeline"):
                    # 只做分词（cws = Chinese Word Segmentation）
                    output = ltp.pipeline([text], tasks=["cws"])

                    # 新版文档里 output.cws 是 List[List[str]]，但也可能是 dict["cws"]
                    cws = getattr(output, "cws", None)
                    if cws is None and isinstance(output, dict):
                        cws = output.get("cws")

                    if cws is None:
                        raise RuntimeError("LTP pipeline output does not contain 'cws' field")

                    tokens = list(cws[0])

                # --- 情况 3：既没有 seg 也没有 pipeline，当作 LTP 不可用 ---
                else:
                    logger.error(
                        "LTP instance has neither 'seg' nor 'pipeline' method; "
                        "falling back to simple regex tokenizer."
                    )
                    tokens = self._regex_tokenize(text)

            except Exception as exc:  # pragma: no cover - 防御性降级
                logger.error(
                    "LTP segmentation failed with %s, falling back to simple regex tokenizer.",
                    exc,
                )
                tokens = self._regex_tokenize(text)
        else:
            tokens = self._regex_tokenize(text)

        if self.cfg.max_tokens > 0 and len(tokens) > self.cfg.max_tokens:
            tokens = tokens[: self.cfg.max_tokens]

        return tokens

    def _regex_tokenize(self, text: str) -> List[str]:
        """
        @brief 基于正则的简易分词实现，提取英文单词、数字和单个 CJK 字符。
            Simple regex-based tokenizer extracting English words, numbers and single CJK chars.
        @param text 清洗后的文本。Cleaned text.
        @return token 列表。List of tokens.
        @note 作为 LTP 不可用时的后备方案，也可用于英文数据集。
            Used as a fallback when LTP is unavailable, and for pure English datasets.
        """
        _TOKEN_PATTERN = re.compile(
            r"""
            [A-Za-z]+(?:'[A-Za-z]+)?   # 英文单词，允许内部一个撇号，如 don't
            | \d+(?:\.\d+)?            # 阿拉伯数字，含简单小数
            | [\u4e00-\u9fff]          # 单个 CJK（中文等）字符
            """,
            re.VERBOSE,
        )

        # 直接从文本中提取合法 token
        raw_tokens = _TOKEN_PATTERN.findall(text)
        tokens = [tok for tok in raw_tokens if tok]

        logger.debug(
            "Regex tokenizer produced %d tokens, first few: %s",
            len(tokens),
            tokens[:10],
        )
        return tokens

    # ------------------------------------------------------------
    # 知识图谱部分：实体与上下文 Entities & Contexts
    # ------------------------------------------------------------
    def _extract_entities_and_contexts(
        self, text: str
    ) -> tuple[List[str], List[List[str]]]:
        """
        @brief 使用知识图谱客户端抽取实体与实体上下文。
               Use KG client to extract entities and their contexts.
        @param text 原始或清洗后的文本。Text (raw or cleaned).
        @return (entities, entity_contexts)：
                - entities: 实体 ID 列表。List of entity IDs.
                - entity_contexts: 与 entities 对齐的一跳邻居 ID 列表。List of neighbor ID lists.
        """
        if not self.cfg.enable_kg or self.kg_client is None:
            # 仅文本模式，返回空实体与上下文
            return [], []

        try:
            raw_entities: Sequence[str] = self.kg_client.link_entities(text)
            entities = list(raw_entities)

            raw_contexts: Sequence[Sequence[str]] = self.kg_client.get_entity_contexts(
                entities
            )
            entity_contexts: List[List[str]] = [list(ctx) for ctx in raw_contexts]

            # 强制对齐：长度不一致时做截断 / 补空
            if len(entity_contexts) < len(entities):
                # 不足则补空列表
                entity_contexts.extend(
                    [[] for _ in range(len(entities) - len(entity_contexts))]
                )
            elif len(entity_contexts) > len(entities):
                entity_contexts = entity_contexts[: len(entities)]

            return entities, entity_contexts
        except Exception as e:  # noqa: BLE001 简化处理
            logger.error(f"KG extraction failed: {e}")
            # 出错时退化为仅文本模式
            return [], []

    # ------------------------------------------------------------
    # 对单个样本进行预处理 Process one sample
    # ------------------------------------------------------------
    def preprocess_sample(self, sample: NewsSample) -> PreprocessedSample:
        """
        @brief 对单条 NewsSample 进行预处理，生成 PreprocessedSample。
               Preprocess a single NewsSample into PreprocessedSample.
        @param sample 输入样本。Input sample.
        @return 预处理后的样本。Preprocessed sample.
        """
        cleaned = self._clean_text(sample.text)
        tokens = self._tokenize(cleaned)
        entities, entity_contexts = self._extract_entities_and_contexts(cleaned)

        return PreprocessedSample(
            id=sample.id,
            tokens=tokens,
            entities=entities,
            entity_contexts=entity_contexts,
            label=sample.label,
        )

    # ------------------------------------------------------------
    # 批量预处理 Process a batch
    # ------------------------------------------------------------
    def preprocess_batch(
        self, samples: Iterable[NewsSample]
    ) -> List[PreprocessedSample]:
        """
        @brief 批量预处理多个样本，常用于训练/推理阶段。
               Preprocess a batch of samples, typically for training/inference.
        @param samples NewsSample 可迭代对象。Iterable of NewsSample.
        @return 预处理后样本列表。List of PreprocessedSample.
        """
        results: List[PreprocessedSample] = []
        for s in samples:
            results.append(self.preprocess_sample(s))
        return results
