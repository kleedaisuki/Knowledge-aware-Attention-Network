"""
@file knowledge_graph.py
@brief 知识图谱访问与实体上下文抽取模块，基于 SPARQL 封装 Wikidata 等 KG。
       Knowledge graph access & entity context extraction utilities based on SPARQL.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import requests
from SPARQLWrapper import SPARQLWrapper, JSON  # type: ignore[import-untyped]

from kan.utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================
# 配置 Configuration
# ============================================================


@dataclass
class KnowledgeGraphConfig:
    """
    @brief 知识图谱客户端配置。Configuration for knowledge graph client.
    @param endpoint_url SPARQL 端点地址，例如 Wikidata endpoint。
           SPARQL endpoint URL, e.g. Wikidata endpoint.
    @param user_agent HTTP User-Agent，用于礼貌访问公共服务。
           HTTP User-Agent header for polite access.
    @param timeout 超时时间（秒）。Request timeout in seconds.
    @param max_neighbors 每个实体最多返回多少一跳邻居。Max number of 1-hop neighbors per entity.
    @param cache_dir 邻居缓存目录（可选），用于减少重复 SPARQL / 搜索请求。
           Optional cache directory for neighbor / search cache to reduce remote calls.
    @param language 查询 label / 搜索时使用的语言代码。
           Language code used when searching for labels.
    """

    endpoint_url: str = "https://query.wikidata.org/sparql"
    user_agent: str = "KAN-KG-Client/0.1 (https://example.org; mailto:you@example.org)"
    timeout: int = 30
    max_neighbors: int = 32
    cache_dir: Optional[str] = "data/kg_cache"
    language: str = "en"


# ============================================================
# 知识图谱客户端 Knowledge Graph Client
# ============================================================


class KnowledgeGraphClient:
    """
    @brief 基于 SPARQL 的知识图谱客户端，提供实体链接与实体上下文抽取接口。
           SPARQL-based knowledge graph client with entity linking & context extraction APIs.
    """

    def __init__(self, cfg: KnowledgeGraphConfig) -> None:
        """
        @brief 初始化知识图谱客户端。Initialize the knowledge graph client.
        @param cfg KnowledgeGraphConfig 配置对象。Configuration object.
        """
        self.cfg = cfg
        self._sparql: Optional[SPARQLWrapper] = None
        self._neighbor_cache: Dict[str, List[str]] = {}
        # 表面形式 → QID 列表的内存缓存
        self._surface_cache: Dict[str, List[str]] = {}

        if self.cfg.cache_dir is not None:
            os.makedirs(self.cfg.cache_dir, exist_ok=True)

            # 预加载磁盘上的搜索缓存（若存在）。
            search_cache_path = Path(self.cfg.cache_dir) / "surface_to_qid.json"
            if search_cache_path.is_file():
                try:
                    with open(search_cache_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        for k, v in data.items():
                            if isinstance(k, str) and isinstance(v, list):
                                self._surface_cache[k] = [str(x) for x in v]
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Failed to load surface cache: {e}")

    # ------------------------------------------------------------
    # 内部工具：获取 SPARQLWrapper 实例
    # ------------------------------------------------------------

    def _get_sparql(self) -> SPARQLWrapper:
        """
        @brief 懒加载创建 SPARQLWrapper 实例。Lazily create SPARQLWrapper instance.
        @return SPARQLWrapper 对象。SPARQLWrapper instance.
        """
        if self._sparql is None:
            sp = SPARQLWrapper(self.cfg.endpoint_url)
            sp.setReturnFormat(JSON)
            sp.setTimeout(self.cfg.timeout)
            sp.addCustomHttpHeader("User-Agent", self.cfg.user_agent)
            self._sparql = sp
        return self._sparql

    # ------------------------------------------------------------
    # 实体链接：从文本 → 实体 ID 列表
    # ------------------------------------------------------------

    def _save_surface_cache(self) -> None:
        """
        @brief 将 surface form → 实体 ID 映射写入磁盘缓存。
               Persist surface-form → entity-id cache to disk.
        """
        if self.cfg.cache_dir is None:
            return
        path = Path(self.cfg.cache_dir) / "surface_to_qid.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._surface_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to save surface cache: {e}")

    def _extract_candidate_spans(self, text: str) -> List[str]:
        """
        @brief 从原始文本中抽取用于 Wikidata 搜索的候选片段。
               Extract candidate surface forms from raw text for Wikidata search.
        @param text 输入文本。Input text.
        @return 候选片段列表（去重后）。List of candidate surface forms (deduplicated).
        @note
            * 英文：尝试匹配连续的大写开头词，如 "Barack Obama"。
            * 其它语言：退化为基于空格的 token，再过滤掉太短的 token。
              For non-English languages, we fall back to space-based tokens and
              drop very short ones.
        """
        candidates: set[str] = set()

        # 1) 已内嵌的 QID 仍然直接识别出来（便于调试）。
        for qid in re.findall(r"\bQ[0-9]+\b", text):
            candidates.add(qid)

        # 2) 英文专名：连续的首字母大写单词序列。
        proper_name_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
        for m in proper_name_pattern.finditer(text):
            span = m.group(1).strip()
            if len(span) >= 3:
                candidates.add(span)

        # 3) Fallback：基于空格的简单 token 化。
        for tok in text.split():
            tok = tok.strip()
            if len(tok) < 3:
                continue
            # 过滤纯标点
            if all(ch in ",.;:!?\"'()[]{}" for ch in tok):
                continue
            candidates.add(tok)

        return list(candidates)

    def _search_wikidata(self, surface: str, limit: int = 1) -> List[str]:
        """
        @brief 使用 Wikidata 的搜索 API，将表面形式映射为一个或多个实体 ID。
               Use Wikidata search API to map a surface form to one or more entity IDs.
        @param surface 表面字符串。Surface form string.
        @param limit 返回的最大实体数量。Maximum number of entity IDs to return.
        @return 匹配到的实体 ID 列表。List of matched entity IDs.
        """
        surface = surface.strip()
        if not surface:
            return []

        # 先查内存缓存
        if surface in self._surface_cache:
            return self._surface_cache[surface][:limit]

        # QID 直接返回
        if re.fullmatch(r"Q[0-9]+", surface):
            self._surface_cache.setdefault(surface, [surface])
            return [surface]

        params = {
            "action": "wbsearchentities",
            "search": surface,
            "language": self.cfg.language,
            "format": "json",
            "limit": str(limit),
        }
        headers = {"User-Agent": self.cfg.user_agent}

        try:
            resp = requests.get(
                "https://www.wikidata.org/w/api.php",
                params=params,
                headers=headers,
                timeout=self.cfg.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Wikidata search failed for {surface!r}: {e}")
            self._surface_cache.setdefault(surface, [])
            return []

        results: List[str] = []
        for item in data.get("search", []):
            qid = item.get("id")
            if isinstance(qid, str) and qid.startswith("Q"):
                results.append(qid)

        self._surface_cache.setdefault(surface, results)
        self._save_surface_cache()

        return results[:limit]

    def link_entities(self, text: str) -> List[str]:
        """
        @brief 利用 Wikidata 搜索 API 进行轻量级实体链接。
               Perform lightweight entity linking using Wikidata search API.
        @param text 输入文本。Input text.
        @return 实体 ID 列表（如 ["Q76", "Q30"]）。List of entity IDs, e.g. ["Q76", "Q30"].
        @note
            这不是一个“工业级”的实体链接器，只是：
            1) 抽取若干候选片段（proper names / tokens）；
            2) 对每个候选调用 Wikidata 搜索；
            3) 合并得到的若干 QID。
            It is NOT a full-fledged EL system, but:
            1) extracts several candidate spans;
            2) queries Wikidata search for each;
            3) merges the resulting QIDs.
        """
        candidates = self._extract_candidate_spans(text)
        entities: List[str] = []

        for surf in candidates:
            qids = self._search_wikidata(surf, limit=1)
            entities.extend(qids)

        # 去重并排序，保持结果稳定。
        unique = sorted(set(entities))
        logger.info(
            "link_entities: found entities %s from text snippet: %r",
            unique,
            text[:80],
        )
        return unique

    # ------------------------------------------------------------
    # 邻居缓存：磁盘读写
    # ------------------------------------------------------------

    def _cache_path(self, entity_id: str) -> Optional[str]:
        """
        @brief 获取实体邻居缓存文件路径（如果启用磁盘缓存）。
               Get cache file path for an entity's neighbors, if disk cache is enabled.
        @param entity_id 实体 ID。Entity ID.
        @return 缓存文件路径或 None。Cache path or None.
        """
        if self.cfg.cache_dir is None:
            return None
        safe_id = re.sub(r"[^A-Za-z0-9]", "_", entity_id)
        return str(Path(self.cfg.cache_dir) / f"neighbors_{safe_id}.json")

    def _load_neighbors_from_disk(self, entity_id: str) -> Optional[List[str]]:
        """
        @brief 从磁盘缓存加载某实体的一跳邻居。
               Load 1-hop neighbors of an entity from disk cache.
        @param entity_id 实体 ID。Entity ID.
        @return 邻居实体 ID 列表或 None。Neighbor entity IDs or None.
        """
        path = self._cache_path(entity_id)
        if path is None or not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [str(x) for x in data]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to load neighbor cache for {entity_id}: {e}")
        return None

    def _save_neighbors_to_disk(self, entity_id: str, neighbors: List[str]) -> None:
        """
        @brief 将实体邻居列表写入磁盘缓存。
               Save neighbor list of an entity to disk cache.
        @param entity_id 实体 ID。Entity ID.
        @param neighbors 邻居实体 ID 列表。List of neighbor entity IDs.
        """
        path = self._cache_path(entity_id)
        if path is None:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(neighbors, f, ensure_ascii=False)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to save neighbor cache for {entity_id}: {e}")

    # ------------------------------------------------------------
    # 对单个实体查询一跳邻居 1-hop neighbors for single entity
    # ------------------------------------------------------------

    def get_neighbors(self, entity_id: str) -> List[str]:
        """
        @brief 获取单个实体在知识图谱中的一跳邻居实体 ID。
               Get 1-hop neighbor entity IDs for a single entity in the KG.
        @param entity_id Wikidata 实体 ID，例如 "Q76"。
               Wikidata entity ID, e.g. "Q76".
        @return 邻居实体 ID 列表（不含自身），长度不超过 max_neighbors。
                List of neighbor entity IDs (excluding self), up to max_neighbors.
        """
        # 内存缓存优先
        if entity_id in self._neighbor_cache:
            return self._neighbor_cache[entity_id]

        # 磁盘缓存其次
        cached = self._load_neighbors_from_disk(entity_id)
        if cached is not None:
            self._neighbor_cache[entity_id] = cached
            return cached

        # 发送 SPARQL 查询
        sparql = self._get_sparql()
        query = f"""
        PREFIX wd: <http://www.wikidata.org/entity/>

        SELECT DISTINCT ?neighbor WHERE {{
          {{
            wd:{entity_id} ?p ?neighbor .
          }}
          UNION
          {{
            ?neighbor ?p wd:{entity_id} .
          }}
          FILTER(isIRI(?neighbor))
        }}
        LIMIT {int(self.cfg.max_neighbors)}
        """
        sparql.setQuery(query)

        try:
            results = sparql.query().convert()
        except Exception as e:  # noqa: BLE001
            logger.error(f"SPARQL query failed for {entity_id}: {e}")
            self._neighbor_cache[entity_id] = []
            return []

        neighbors: List[str] = []
        for row in results.get("results", {}).get("bindings", []):
            uri = row.get("neighbor", {}).get("value")
            if not uri:
                continue
            # 形如 http://www.wikidata.org/entity/Q76 → Q76
            qid = uri.rsplit("/", 1)[-1]
            if qid != entity_id:
                neighbors.append(qid)

        neighbors = sorted(set(neighbors))
        self._neighbor_cache[entity_id] = neighbors
        self._save_neighbors_to_disk(entity_id, neighbors)

        logger.info(f"get_neighbors: {entity_id} -> {len(neighbors)} neighbors")
        return neighbors

    # ------------------------------------------------------------
    # 批量获取实体上下文 Entity contexts for a list of entities
    # ------------------------------------------------------------

    def get_entity_contexts(self, entities: Sequence[str]) -> List[List[str]]:
        """
        @brief 获取一组实体各自的一跳邻居列表（实体上下文）。
               Get 1-hop neighbors (entity contexts) for a list of entities.
        @param entities Wikidata 实体 ID 序列。Sequence of Wikidata entity IDs.
        @return 每个实体对应的邻居 ID 列表，顺序与输入对齐。
                List of neighbor ID lists, aligned with input order.
        """
        contexts: List[List[str]] = []
        for eid in entities:
            neighbors = self.get_neighbors(eid)
            contexts.append(neighbors)
        return contexts
