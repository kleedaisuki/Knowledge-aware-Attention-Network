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
from typing import Dict, List, Optional, Sequence

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
    @param cache_dir 邻居缓存目录（可选），用于减少重复 SPARQL 查询。
           Optional cache directory for neighbor results to reduce SPARQL calls.
    @param language 未来如需查询 label，可指定语言；当前逻辑暂未使用。
           Language code for labels if needed in future (currently unused).
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

        if self.cfg.cache_dir is not None:
            os.makedirs(self.cfg.cache_dir, exist_ok=True)

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

    def link_entities(self, text: str) -> List[str]:
        """
        @brief 极简实体链接：从文本中抽取形如 'Q12345' 的 Wikidata 实体 ID。
               Minimal entity linking: extract Wikidata-like IDs 'Q12345' from text.
        @param text 输入文本。Input text.
        @return 实体 ID 列表（如 ["Q76", "Q30"]）。List of entity IDs, e.g. ["Q76", "Q30"].
        @note 生产环境建议替换为真正的实体链接组件（如 TagMe / BLINK），但接口保持不变。
              In production, replace this with a real entity linker, keeping the same API.
        """
        # 查找所有形如 Q123、Q42 的 token
        candidates = set(re.findall(r"\bQ[0-9]+\b", text))
        entities = sorted(candidates)
        logger.debug(
            f"link_entities: found entities {entities} from text snippet: {text[:50]!r}"
        )
        return entities

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
        safe_id = entity_id.replace("/", "_")
        return os.path.join(self.cfg.cache_dir, f"{safe_id}.json")

    def _load_neighbors_from_disk(self, entity_id: str) -> Optional[List[str]]:
        """
        @brief 从磁盘缓存读取实体的一跳邻居列表。
               Load 1-hop neighbors of an entity from disk cache.
        @param entity_id 实体 ID。Entity ID.
        @return 邻居列表或 None。Neighbor list or None if not cached.
        """
        path = self._cache_path(entity_id)
        if path is None or not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [str(x) for x in data]
        except Exception as e:  # noqa: BLE001
            logger.warn(f"Failed to load neighbor cache for {entity_id}: {e}")
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
            logger.warn(f"Failed to save neighbor cache for {entity_id}: {e}")

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
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>

        SELECT DISTINCT ?neighbor WHERE {{
          {{
            wd:{entity_id} ?p ?neighbor .
          }}
          UNION
          {{
            ?neighbor ?p wd:{entity_id} .
          }}
          FILTER(STRSTARTS(STR(?neighbor), "http://www.wikidata.org/entity/"))
        }}
        LIMIT {self.cfg.max_neighbors}
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

        logger.debug(f"get_neighbors: {entity_id} -> {len(neighbors)} neighbors")
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
