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

logger = get_logger(__name__)

from ltp import LTP
import jieba  # type: ignore[import-untyped]


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
    @param tokenizer 分词器类型：
           - "jieba"  : 使用 jieba 中文分词；
           - "ltp"    : 使用 LTP 中文分词；
           - "simple" : 使用简易正则分词（等价于 "regex"）；
           - "regex"  : 使用简易正则分词。
           Type of tokenizer:
           - "jieba"  : jieba-based Chinese word segmentation;
           - "ltp"    : LTP-based Chinese segmentation;
           - "simple" : simple regex tokenizer (alias of "regex");
           - "regex"  : simple regex tokenizer.
    @param ltp_model_kwargs LTP 初始化参数（可选），例如模型名和 cache 目录。
           Optional kwargs for initializing LTP, e.g. model name and cache dir.
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
        @param kg_client 知识图谱客户端，需提供实体链接与实体上下文接口。
               Knowledge graph client providing entity linking & context APIs.
        @note kg_client 预期接口（新接口优先，旧接口作为兼容）：
              - link_entities_from_tokens(tokens: Sequence[str]) -> Sequence[str]
              - get_entity_contexts(entities: Sequence[str]) -> Sequence[Sequence[str]]
              - （兼容旧版）link_entities(text: str) -> Sequence[str]
        """
        self.cfg = cfg
        self.kg_client = kg_client

        # 统一的 tokenizer 类型标记（路由中枢）。
        # Unified tokenizer type tag (routing center).
        raw_tok = getattr(self.cfg, "tokenizer", "ltp") or "ltp"
        tok = str(raw_tok).lower()

        self._tokenizer_type: str = "regex"
        self._ltp: Any = None

        if tok == "ltp":
            # 允许通过 config.ltp_model_kwargs 追加更多底层模型参数
            ltp_kwargs = getattr(self.cfg, "ltp_model_kwargs", None)

            try:
                kwargs = ltp_kwargs if isinstance(ltp_kwargs, dict) else {}
                self._ltp = LTP(**kwargs)
                self._tokenizer_type = "ltp"

                def _try_get(d: Dict, key: Any, default: Any = None):
                    if not isinstance(d, dict):
                        return default
                    value = d.get(key)
                    return value if value is not None else default

                logger.info(
                    "Preprocessor: initialized LTP tokenizer "
                    "(model=%s, cache_dir=%s, local_files_only=%s).",
                    _try_get(
                        ltp_kwargs or {},
                        "pretrained_model_name_or_path",
                        "LTP/small",
                    ),
                    _try_get(ltp_kwargs or {}, "cache_dir", None),
                    _try_get(ltp_kwargs or {}, "local_files_only", False),
                )
            except Exception as exc:  # pragma: no cover - 防御性降级
                logger.error(
                    "Preprocessor: failed to initialize LTP with kwargs %s, "
                    "falling back to regex tokenizer. Error: %s",
                    ltp_kwargs,
                    exc,
                )
                self._ltp = None
                self._tokenizer_type = "regex"

        elif tok == "jieba":
            self._tokenizer_type = "jieba"
            logger.info("Preprocessor: using jieba tokenizer.")

        elif tok in ("simple", "regex"):
            self._tokenizer_type = "regex"
            logger.info(
                "Preprocessor: using simple regex-based tokenizer (tokenizer=%s).", tok
            )
        else:
            # 未知配置统一回退到 regex，避免爆炸。
            # Unknown tokenizer falls back to regex to avoid crash.
            self._tokenizer_type = "regex"
            logger.warning(
                "Preprocessor: unknown tokenizer=%r, falling back to regex tokenizer.",
                tok,
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

        logger.debug("Clean text from: [%s...] to [%s...]", orig[:30], text[:30])
        return text

    def _tokenize_with_ltp(self, text: str) -> List[str]:
        """
        @brief 使用 LTP 对文本进行分词的后端实现。
               Backend implementation of tokenization using LTP.
        @param text 清洗后的文本。Cleaned text.
        @return token 列表。List of tokens.
        """
        if self._ltp is None:
            raise RuntimeError("LTP instance is not initialized.")

        ltp = self._ltp

        # --- 情况 1：老版 LTP（有 seg 方法） ---
        if hasattr(ltp, "seg"):
            seg_result = ltp.seg([text])
            # seg(...) 既可能直接返回 segments，也可能返回 (segments, hidden)
            if isinstance(seg_result, tuple):
                segments = seg_result[0]
            else:
                segments = seg_result
            tokens = list(segments[0])
            return tokens

        # --- 情况 2：新版 LTP（Transformers 版，使用 pipeline 接口） ---
        if hasattr(ltp, "pipeline"):
            # 只做分词（cws = Chinese Word Segmentation）
            output = ltp.pipeline([text], tasks=["cws"])

            # 新版文档里 output.cws 是 List[List[str]]，但也可能是 dict["cws"]
            cws = getattr(output, "cws", None)
            if cws is None and isinstance(output, dict):
                cws = output.get("cws")

            if cws is None:
                raise RuntimeError("LTP pipeline output does not contain 'cws' field.")

            tokens = list(cws[0])
            return tokens

        # --- 情况 3：既没有 seg 也没有 pipeline，当作 LTP 不可用 ---
        raise RuntimeError(
            "LTP instance has neither 'seg' nor 'pipeline' method; "
            "cannot perform LTP tokenization."
        )

    def _tokenize_with_jieba(self, text: str) -> List[str]:
        """
        @brief 使用 jieba 对文本进行分词的后端实现。
               Backend implementation of tokenization using jieba.
        @param text 清洗后的文本。Cleaned text.
        @return token 列表。List of tokens.
        @note
            - 仅在安装了 jieba 且 tokenizer='jieba' 时启用。
              Enabled only when jieba is installed and tokenizer='jieba'.
        """
        # jieba.lcut 返回一个 token 列表，这里做简单 strip 过滤空白
        tokens = [tok.strip() for tok in jieba.lcut(text) if tok and tok.strip()]
        logger.debug(
            "jieba tokenizer produced %d tokens, first few: %s",
            len(tokens),
            tokens[:10],
        )
        return tokens

    def _regex_tokenize(self, text: str) -> List[str]:
        """
        @brief 基于正则的简易分词实现，提取英文单词、数字和单个 CJK 字符。
               Simple regex-based tokenizer extracting English words, numbers and single CJK chars.
        @param text 清洗后的文本。Cleaned text.
        @return token 列表。List of tokens.
        @note 作为 LTP / jieba 不可用时的后备方案，也可用于英文数据集。
              Used as a fallback when LTP/jieba is unavailable, and for pure English datasets.
        """
        _TOKEN_PATTERN = re.compile(
            r"""
            [A-Za-z]+(?:'[A-Za-z]+)?   # 英文单词，允许内部一个撇号，如 don't
            | \d+(?:\.\d+)?            # 阿拉伯数字，含简单小数
            | [\u4e00-\u9fff]          # 单个 CJK（中文等）字符
            """,
            re.VERBOSE,
        )

        raw_tokens = _TOKEN_PATTERN.findall(text)
        tokens = [tok for tok in raw_tokens if tok]

        logger.debug(
            "Regex tokenizer produced %d tokens, first few: %s",
            len(tokens),
            tokens[:10],
        )
        return tokens

    def _tokenize(self, text: str) -> List[str]:
        """
        @brief 统一的 token 化入口，根据配置路由到具体实现。
               Unified tokenization entry, routing to concrete backend by config.
        @param text 清洗后的文本。Cleaned text.
        @return token 列表（如配置指定，还会在此处做 max_tokens 截断）。
                List of tokens (also truncated here if max_tokens > 0).
        @note
            - 路由逻辑集中在此处，方便以后扩展新的分词后端；
              Routing logic is centralized here for easier extension.
            - 各后端实现仅关心“如何把 text → tokens”，上层接口保持稳定。
              Backend implementations only care about mapping text → tokens.
        """
        tokens: List[str]

        if self._tokenizer_type == "jieba":
            try:
                tokens = self._tokenize_with_jieba(text)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "jieba tokenization failed with %s, falling back to regex tokenizer.",
                    exc,
                )
                tokens = self._regex_tokenize(text)

        elif self._tokenizer_type == "ltp":
            try:
                tokens = self._tokenize_with_ltp(text)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "LTP tokenization failed with %s, falling back to regex tokenizer.",
                    exc,
                )
                tokens = self._regex_tokenize(text)

        else:
            # 默认 / 回退：正则分词
            tokens = self._regex_tokenize(text)

        # 统一在这里处理 max_tokens 截断，保证行为可预期。
        # Handle max_tokens truncation here to keep behavior consistent.
        if self.cfg.max_tokens > 0 and len(tokens) > self.cfg.max_tokens:
            tokens = tokens[: self.cfg.max_tokens]

        return tokens

    # ------------------------------------------------------------
    # 知识图谱部分：实体与上下文 Entities & Contexts
    # ------------------------------------------------------------
    def _extract_entities_and_contexts(
        self, text: str, tokens: Sequence[str]
    ) -> tuple[List[str], List[List[str]]]:
        """
        @brief 使用知识图谱客户端抽取实体与实体上下文。
               Use KG client to extract entities and their contexts.
        @param text 原始或清洗后的文本（仅用于日志/回退）。Text (raw or cleaned), used for logging/fallback.
        @param tokens 预处理得到的 token 序列。Token sequence produced by preprocessing.
        @return (entities, entity_contexts)：
                - entities: 实体 ID 列表。List of entity IDs.
                - entity_contexts: 与 entities 对齐的一跳邻居 ID 列表。List of neighbor ID lists.
        """
        if not self.cfg.enable_kg or self.kg_client is None:
            # 仅文本模式，返回空实体与上下文
            return [], []

        try:
            # ✅ 优先使用基于 token 的实体链接
            if hasattr(self.kg_client, "link_entities_from_tokens"):
                raw_entities: Sequence[str] = self.kg_client.link_entities_from_tokens(
                    tokens
                )
            else:
                # 兼容旧实现：退回基于文本的接口
                raw_entities = self.kg_client.link_entities(text)

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
            logger.error("KG extraction failed: %s", e)
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

        # ✅ 把 token 序列传给 KG，实体链接完全基于分好词的结果
        entities, entity_contexts = self._extract_entities_and_contexts(cleaned, tokens)

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
