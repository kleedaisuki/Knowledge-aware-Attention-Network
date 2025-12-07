"""
@file preprocessing.py
@brief 文本与知识图谱联合预处理模块，将原始样本映射为 (S, E, EC) 形式。
       Text & knowledge-graph joint preprocessing module, mapping raw samples to (S, E, EC).
"""

from __future__ import annotations

import re
import asyncio
from dataclasses import dataclass
from typing import List, Optional, Any, Sequence, Iterable, Dict, Tuple

from kan.utils.logging import get_logger
from kan.data.datasets import NewsSample
from kan.data.knowledge_graph import KnowledgeGraphClient

from ltp import LTP  # type: ignore[import-untyped]
import jieba  # type: ignore[import-untyped]

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
    tokenizer: str = "jieba"
    ltp_model_kwargs: Optional[Dict[str, Any]] = None


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
        self, cfg: PreprocessConfig, kg_client: Optional[KnowledgeGraphClient] = None
    ) -> None:
        """
        @brief 初始化预处理器。Initialize preprocessor.
        @param cfg PreprocessConfig 配置对象。Preprocess configuration object.
        @param kg_client 知识图谱客户端，需提供实体链接与实体上下文接口。
               Knowledge graph client providing entity linking & context APIs.
        @note kg_client 预期接口（新接口优先，旧接口作为兼容）：
              - async alink_entities_from_tokens(tokens: Sequence[str]) -> List[str]
              - async aget_entity_contexts(entities: Sequence[str]) -> List[List[str]]
              - link_entities_from_tokens(tokens: Sequence[str]) -> List[str]
              - get_entity_contexts(entities: Sequence[str]) -> List[List[str]]
              - （兼容旧版）link_entities(text: str) -> List[str]
        """
        self.cfg = cfg
        self.kg_client = kg_client

        # 统一的 tokenizer 类型标记（路由中枢）。
        raw_tok = getattr(self.cfg, "tokenizer", "ltp") or "ltp"
        tok = str(raw_tok).lower()

        self._tokenizer_type: str = "regex"
        self._ltp: Any = None

        if tok == "ltp":
            ltp_kwargs = getattr(self.cfg, "ltp_model_kwargs", None)

            try:
                kwargs = ltp_kwargs if isinstance(ltp_kwargs, dict) else {}
                self._ltp = LTP(**kwargs)
                self._tokenizer_type = "ltp"

                def _try_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
                    if not isinstance(d, dict):
                        return default
                    value = d.get(key)
                    return default if value is None else value

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
            except Exception as exc:  # pragma: no cover
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
        """
        orig = text

        if self.cfg.do_lowercase:
            text = text.lower()

        if self.cfg.remove_urls:
            text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        if self.cfg.remove_user_mentions:
            text = re.sub(r"@\w+", " ", text)

        text = re.sub(r"\s+", " ", text).strip()

        logger.debug("Clean text from: [%s...] to [%s...]", orig[:30], text[:30])
        return text

    def _tokenize_with_ltp(self, text: str) -> List[str]:
        """使用 LTP 分词。Tokenize with LTP."""
        if self._ltp is None:
            raise RuntimeError("LTP instance is not initialized.")

        ltp = self._ltp

        if hasattr(ltp, "seg"):
            seg_result = ltp.seg([text])
            if isinstance(seg_result, tuple):
                segments = seg_result[0]
            else:
                segments = seg_result
            return list(segments[0])

        if hasattr(ltp, "pipeline"):
            output = ltp.pipeline([text], tasks=["cws"])
            cws = getattr(output, "cws", None)
            if cws is None and isinstance(output, dict):
                cws = output.get("cws")
            if cws is None:
                raise RuntimeError("LTP pipeline output does not contain 'cws' field.")
            return list(cws[0])

        raise RuntimeError(
            "LTP instance has neither 'seg' nor 'pipeline' method; cannot tokenize."
        )

    def _tokenize_with_jieba(self, text: str) -> List[str]:
        """使用 jieba 分词。Tokenize with jieba."""
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
        """
        _TOKEN_PATTERN = re.compile(
            r"""
            [A-Za-z]+(?:'[A-Za-z]+)?   # 英文单词，允许内部一个撇号，如 don't
            | \d+(?:\.\d+)?            # 数字，含简单小数
            | [\u4e00-\u9fff]          # 单个 CJK 字符
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
        @brief 统一 token 化入口，根据配置路由到具体实现。
               Unified tokenization entry.
        """
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
            tokens = self._regex_tokenize(text)

        if self.cfg.max_tokens > 0 and len(tokens) > self.cfg.max_tokens:
            tokens = tokens[: self.cfg.max_tokens]

        return tokens

    # ------------------------------------------------------------
    # KG：核心异步实现（优先异步 API）
    # KG: core async impl (prefer async APIs)
    # ------------------------------------------------------------
    async def _kg_link_and_context(
        self, text: str, tokens: Sequence[str]
    ) -> Tuple[List[str], List[List[str]]]:
        """
        @brief 使用知识图谱客户端抽取实体与实体上下文（异步主路径）。
               Use KG client to extract entities and contexts (async primary path).
        """
        if not self.cfg.enable_kg or self.kg_client is None:
            return [], []

        kg = self.kg_client
        loop = asyncio.get_running_loop()

        # ---------- 实体链接：优先异步接口 ----------
        entities: List[str]

        if hasattr(kg, "alink_entities_from_tokens"):
            # 首选：直接用异步 token-based 接口
            entities = await kg.alink_entities_from_tokens(tokens)  # type: ignore[attr-defined]
        elif hasattr(kg, "alink_entities"):
            # 次选：异步 text-based 接口
            entities = await kg.alink_entities(text)  # type: ignore[attr-defined]
        elif hasattr(kg, "link_entities_from_tokens"):
            # 再退：同步 token-based，扔到线程池里
            def _link_sync_tokens() -> List[str]:
                return list(kg.link_entities_from_tokens(tokens))  # type: ignore[arg-type]

            entities = await loop.run_in_executor(None, _link_sync_tokens)
        elif hasattr(kg, "link_entities"):
            # 最后：同步 text-based
            def _link_sync_text() -> List[str]:
                return list(kg.link_entities(text))  # type: ignore[arg-type]

            entities = await loop.run_in_executor(None, _link_sync_text)
        else:
            logger.warning(
                "KG client has no entity-linking API, return empty entities for now."
            )
            return [], []

        if not entities:
            return [], []

        # ---------- 实体上下文：优先异步接口 ----------
        contexts: List[List[str]]

        if hasattr(kg, "aget_entity_contexts"):
            contexts = await kg.aget_entity_contexts(entities)  # type: ignore[attr-defined]
        elif hasattr(kg, "get_entity_contexts"):

            def _ctx_sync() -> List[List[str]]:
                ctxs = kg.get_entity_contexts(entities)
                return [list(c) for c in ctxs]

            contexts = await loop.run_in_executor(None, _ctx_sync)
        else:
            logger.warning(
                "KG client has no get_entity_contexts API, contexts will be empty."
            )
            contexts = [[] for _ in entities]

        # 对齐长度，保证一一对应
        if len(contexts) < len(entities):
            contexts.extend([[] for _ in range(len(entities) - len(contexts))])
        elif len(contexts) > len(entities):
            contexts = contexts[: len(entities)]

        return list(entities), contexts

    def _kg_link_and_context_sync(
        self, text: str, tokens: Sequence[str]
    ) -> Tuple[List[str], List[List[str]]]:
        """
        @brief 同步 fallback：在无法使用 asyncio.run 的环境中使用。
               Sync fallback used when asyncio.run cannot be used.
        """
        if not self.cfg.enable_kg or self.kg_client is None:
            return [], []

        kg = self.kg_client

        # 这里就老老实实用同步 API，不再搞一层 event loop。
        if hasattr(kg, "link_entities_from_tokens"):
            entities = list(kg.link_entities_from_tokens(tokens))  # type: ignore[arg-type]
        elif hasattr(kg, "link_entities"):
            entities = list(kg.link_entities(text))  # type: ignore[arg-type]
        else:
            return [], []

        if not entities:
            return [], []

        if hasattr(kg, "get_entity_contexts"):
            ctxs = kg.get_entity_contexts(entities)
            contexts = [list(c) for c in ctxs]
        else:
            contexts = [[] for _ in entities]

        if len(contexts) < len(entities):
            contexts.extend([[] for _ in range(len(entities) - len(contexts))])
        elif len(contexts) > len(entities):
            contexts = contexts[: len(entities)]

        return entities, contexts

    # ------------------------------------------------------------
    # 单样本预处理（同步 API，内部优先异步 KG）
    # Single-sample preprocess (sync API, async KG inside)
    # ------------------------------------------------------------
    def preprocess_sample(self, sample: NewsSample) -> PreprocessedSample:
        """
        @brief 对单条 NewsSample 进行预处理（同步 API）。
               Preprocess a single NewsSample (sync API).
        """
        cleaned = self._clean_text(sample.text)
        tokens = self._tokenize(cleaned)

        # KG disabled / no client: text-only
        if not self.cfg.enable_kg or self.kg_client is None:
            return PreprocessedSample(
                id=sample.id,
                tokens=tokens,
                entities=[],
                entity_contexts=[],
                label=sample.label,
            )

        # 优先使用 asyncio.run 驱动异步 KG
        try:

            async def _runner() -> Tuple[List[str], List[List[str]]]:
                return await self._kg_link_and_context(cleaned, tokens)

            entities, contexts = asyncio.run(_runner())
        except RuntimeError as exc:
            # 如果已经有事件循环（如 notebook / web 服务器），不能用 asyncio.run
            # 这时退回到同步 KG fallback，至少不炸。
            logger.warning(
                "preprocess_sample: asyncio.run failed (%s), fallback to sync KG.",
                exc,
            )
            entities, contexts = self._kg_link_and_context_sync(cleaned, tokens)

        return PreprocessedSample(
            id=sample.id,
            tokens=tokens,
            entities=entities,
            entity_contexts=contexts,
            label=sample.label,
        )

    # ------------------------------------------------------------
    # 批处理预处理（同步 API，批量并发异步 KG）
    # Batch preprocess (sync API, batched async KG)
    # ------------------------------------------------------------
    def preprocess_batch(
        self, samples: Iterable[NewsSample]
    ) -> List[PreprocessedSample]:
        """
        @brief 批量预处理多个样本（同步 API）。
               Preprocess a batch of samples synchronously.
        """
        sample_list = list(samples)
        if not sample_list:
            return []

        # 纯文本模式：不走 KG，简单 for-loop 即可
        if not self.cfg.enable_kg or self.kg_client is None:
            out: List[PreprocessedSample] = []
            for s in sample_list:
                cleaned = self._clean_text(s.text)
                tokens = self._tokenize(cleaned)
                out.append(
                    PreprocessedSample(
                        id=s.id,
                        tokens=tokens,
                        entities=[],
                        entity_contexts=[],
                        label=s.label,
                    )
                )
            return out

        # 清洗 & 分词：同步完成
        cleaned_list: List[str] = []
        token_list: List[List[str]] = []
        id_list: List[int] = []
        label_list: List[Optional[int]] = []

        for s in sample_list:
            cleaned = self._clean_text(s.text)
            tokens = self._tokenize(cleaned)
            cleaned_list.append(cleaned)
            token_list.append(tokens)
            id_list.append(s.id)
            label_list.append(s.label)

        async def _run_batch_kg() -> List[PreprocessedSample]:
            # 为每个样本创建异步 KG 任务（**这里是真正的批量并发**）
            tasks = [
                self._kg_link_and_context(text, toks)
                for text, toks in zip(cleaned_list, token_list)
            ]
            kg_results = await asyncio.gather(*tasks, return_exceptions=False)

            out: List[PreprocessedSample] = []
            for sid, label, tokens, (entities, ctxs) in zip(
                id_list, label_list, token_list, kg_results
            ):
                out.append(
                    PreprocessedSample(
                        id=sid,
                        tokens=tokens,
                        entities=entities,
                        entity_contexts=ctxs,
                        label=label,
                    )
                )
            return out

        try:
            return asyncio.run(_run_batch_kg())
        except RuntimeError as exc:
            # 已存在事件循环时退回逐样本同步；虽然慢，但保证行为正确。
            logger.warning(
                "preprocess_batch: asyncio.run failed (%s), fallback to per-sample sync.",
                exc,
            )
            results: List[PreprocessedSample] = []
            for s in sample_list:
                results.append(self.preprocess_sample(s))
            return results
