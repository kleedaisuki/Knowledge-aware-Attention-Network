"""
@file knowledge_graph.py
@brief 知识图谱访问与实体上下文抽取模块，基于 SPARQL 封装 Wikidata 等知识图谱。
       Knowledge graph access & entity context extraction utilities based on SPARQL (e.g. Wikidata).
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import asyncio
from concurrent.futures import ThreadPoolExecutor, Future

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
    @brief 知识图谱客户端配置。Configuration for the knowledge graph client.
    @param endpoint_url SPARQL 端点地址，例如 Wikidata endpoint。
           SPARQL endpoint URL, e.g. Wikidata public endpoint.
    @param user_agent HTTP User-Agent，用于礼貌访问公共服务。
           HTTP User-Agent header used for polite access to public services.
    @param timeout 超时时间（秒）。Request timeout in seconds.
    @param max_neighbors 每个实体最多返回多少一跳邻居。Max number of 1-hop neighbors per entity.
    @param cache_dir 邻居与表面形式缓存目录（可选），用于减少重复远程请求。
           Optional cache directory for neighbor / surface caches to reduce remote calls.
    @param language 查询 label / 搜索时使用的语言代码（如 "zh"、"en"）。
           Language code used when searching for labels, e.g. "zh", "en".
    @param max_workers 线程池最大工作线程数，用于并发网络 I/O。
           Max worker threads for the internal thread pool used for concurrent network I/O.
    """

    endpoint_url: str = "https://query.wikidata.org/sparql"
    user_agent: str = "KAN-KG-Client/0.1"
    timeout: int = 30
    max_neighbors: int = 32
    cache_dir: Optional[str] = "data/kg_cache"
    language: str = "zh"
    max_workers: int = 20
    neighbor_query_mode: str = "full"


# ============================================================
# 客户端实现 Client implementation
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

        # 为支持多线程并发查询，不再跨线程复用同一个 SPARQLWrapper 实例。
        # To support multi-threaded queries, we no longer reuse a single SPARQLWrapper instance.
        self._sparql: Optional[SPARQLWrapper] = (
            None  # kept only for potential future use
        )

        # 邻居缓存：实体 ID -> 邻居 ID 列表（内存级）。
        # Neighbor cache: entity ID -> list of neighbor IDs (in-memory).
        self._neighbor_cache: Dict[str, List[str]] = {}

        # 表面形式缓存：surface string -> QID 列表（内存级）。
        # Surface cache: surface form -> list of QIDs (in-memory).
        self._surface_cache: Dict[str, List[str]] = {}

        # 缓存锁：保证多线程读写安全。
        # Locks for caches to ensure thread-safe access in multi-threaded environments.
        self._surface_lock = threading.Lock()
        self._neighbor_lock = threading.Lock()

        # 网络 I/O 线程池（惰性创建）。
        # Thread pool for network I/O (created lazily).
        self._executor: Optional[ThreadPoolExecutor] = None

        # 缓存 SPARQLWrapper 的线程局部存储（每个线程一个实例，既线程安全又能复用连接）
        # Thread-local storage for SPARQLWrapper instances (one per thread).
        self._sparql_local = threading.local()

        # 准备缓存目录并尝试加载表面形式缓存。
        # Prepare cache directory and try to load surface cache from disk.
        if self.cfg.cache_dir is not None:
            cache_path = Path(self.cfg.cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            self._load_surface_cache()

    # ------------------------------------------------------------
    # 资源管理：线程池与 SPARQL
    # ------------------------------------------------------------

    def _get_executor(self) -> ThreadPoolExecutor:
        """
        @brief 获取（或惰性创建）内部线程池，用于并发网络 I/O。
               Get (or lazily create) the internal thread pool for concurrent network I/O.
        @return ThreadPoolExecutor 实例。ThreadPoolExecutor instance.
        """
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.cfg.max_workers,
                thread_name_prefix="kg-io",
            )
        return self._executor

    def close(self) -> None:
        """
        @brief 显式关闭内部线程池等资源。
               Explicitly shutdown the internal thread pool and related resources.
        @note
            - 建议在长时间训练结束后调用，以加快进程退出。
              It is recommended to call this after long-running training to accelerate process exit.
        """
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        """
        @brief 析构时尝试回收资源（尽力而为）。
               Best-effort resource cleanup when the object is being destroyed.
        """
        try:
            self.close()
        except Exception:  # noqa: BLE001
            # 在析构中绝不抛出异常。
            # Never raise exceptions from __del__.
            pass

    def _get_sparql(self) -> SPARQLWrapper:
        """
        @brief 获取当前线程专属的 SPARQLWrapper 实例（线程安全且复用连接）。
               Get a per-thread SPARQLWrapper instance (thread-safe and reusing connections).
        @return SPARQLWrapper 对象。SPARQLWrapper instance.
        """
        local = self._sparql_local
        sp = getattr(local, "sparql", None)
        if sp is None:
            sp = SPARQLWrapper(self.cfg.endpoint_url)
            sp.setReturnFormat(JSON)
            sp.setTimeout(self.cfg.timeout)
            sp.addCustomHttpHeader("User-Agent", self.cfg.user_agent)
            # 如果 endpoint 支持 keep-alive，复用一个实例可以尽量复用底层连接
            # Reuse the underlying HTTP connection if the endpoint supports keep-alive.
            local.sparql = sp
        return sp

    # ------------------------------------------------------------
    # 缓存工具：路径与持久化
    # ------------------------------------------------------------

    def _cache_path(self, name: str) -> Path:
        """
        @brief 构造缓存文件路径。Build the cache file path.
        @param name 缓存文件名（不含目录）。Cache file name (without directory).
        @return 完整 Path 对象。Full Path object for the cache file.
        """
        if self.cfg.cache_dir is None:
            # 没有显式 cache_dir 时使用当前工作目录下的临时路径。
            # Use current working directory as a fallback when cache_dir is None.
            return Path(name)
        return Path(self.cfg.cache_dir) / name

    def _load_neighbors_from_disk(self, entity_id: str) -> Optional[List[str]]:
        """
        @brief 从磁盘加载单个实体邻居缓存（如果存在）。
               Load cached neighbors for a single entity from disk if present.
        @param entity_id 实体 ID。Entity ID.
        @return 邻居 ID 列表或 None。List of neighbor IDs or None.
        """
        if self.cfg.cache_dir is None:
            return None
        path = self._cache_path(f"{entity_id}.neighbors.json")
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [str(x) for x in data]
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load neighbor cache for %s: %s", entity_id, e)
        return None

    def _save_neighbors_to_disk(self, entity_id: str, neighbors: Sequence[str]) -> None:
        """
        @brief 将单个实体的邻居列表保存到磁盘缓存。
               Save the neighbor list for a single entity to disk.
        @param entity_id 实体 ID。Entity ID.
        @param neighbors 邻居 ID 序列。Neighbor ID sequence.
        """
        if self.cfg.cache_dir is None:
            return
        path = self._cache_path(f"{entity_id}.neighbors.json")
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(list(neighbors), f, ensure_ascii=False)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to save neighbor cache for %s: %s", entity_id, e)

    def _load_surface_cache(self) -> None:
        """
        @brief 尝试从磁盘加载表面形式缓存。
               Try to load the surface-form cache from disk.
        """
        if self.cfg.cache_dir is None:
            return
        path = self._cache_path("surface_cache.json")
        if not path.exists():
            return
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, Mapping):
                new_cache: Dict[str, List[str]] = {}
                for k, v in data.items():
                    if isinstance(v, list):
                        new_cache[str(k)] = [str(x) for x in v]
                # 用锁保护赋值，避免并发下直接替换引用导致奇怪问题
                # Assign under lock to avoid races.
                with self._surface_lock:
                    self._surface_cache = new_cache
                logger.info(
                    "Loaded surface cache with %d entries", len(self._surface_cache)
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load surface cache: %s", e)

    def _save_surface_cache(self) -> None:
        """
        @brief 将当前表面形式缓存保存到磁盘。
               Persist the current surface-form cache to disk.
        """
        if self.cfg.cache_dir is None:
            return
        path = self._cache_path("surface_cache.json")
        try:
            # 🔒 在锁内拷贝一份快照，避免遍历过程中字典被其他线程修改。
            # Take a snapshot under the lock to avoid "dictionary changed size during iteration".
            with self._surface_lock:
                data = dict(self._surface_cache)

            # 在锁外执行磁盘写入，减少锁持有时间。
            # Do disk I/O outside the lock to minimize lock contention.
            with path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to save surface cache: %s", e)

    def _build_neighbor_query(self, entity_id: str) -> str:
        """
        @brief 根据配置构造邻居查询用的 SPARQL。
               Build SPARQL query for neighbor retrieval according to config.
        @param entity_id Wikidata 实体 ID，例如 "Q76"。
               Wikidata entity ID, e.g. "Q76".
        @return SPARQL 查询字符串。SPARQL query string.
        """
        mode = (self.cfg.neighbor_query_mode or "full").lower()

        if mode == "direct":
            # 只使用 direct properties (wdt:)，只查出边：wd:Q ?p ?neighbor
            # This is much lighter than scanning both directions and all property namespaces.
            return f"""
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>

            SELECT DISTINCT ?neighbor WHERE {{
              wd:{entity_id} ?p ?neighbor .
              FILTER(isIRI(?neighbor))
              FILTER(STRSTARTS(STR(?p), STR(wdt:)))
            }}
            LIMIT {int(self.cfg.max_neighbors)}
            """

        # 默认模式：保持原有语义（双向 + 任意谓词）
        return f"""
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

    # ------------------------------------------------------------
    # Wikidata 搜索接口（同步 & 异步）
    # ------------------------------------------------------------

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

        # 先查内存缓存（加锁）。
        # First consult in-memory cache (with lock).
        with self._surface_lock:
            cached = self._surface_cache.get(surface)
        if cached is not None:
            return cached[:limit]

        # 构造 Wikidata wbsearchentities 请求。
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": self.cfg.language,
            "search": surface,
            "limit": max(1, int(limit)),
        }
        url = "https://www.wikidata.org/w/api.php"

        try:
            resp = requests.get(
                url,
                params=params,
                timeout=self.cfg.timeout,
                headers={"User-Agent": self.cfg.user_agent},
            )
            resp.raise_for_status()
            data: Dict[str, Any] = resp.json()
        except Exception as e:  # noqa: BLE001
            logger.warning("Wikidata search failed for %r: %s", surface, e)
            # 失败时仍然初始化空列表，避免重复打同一个失败请求。
            with self._surface_lock:
                self._surface_cache.setdefault(surface, [])
            return []

        results: List[str] = []
        for item in data.get("search", []):
            qid = item.get("id")
            if isinstance(qid, str) and qid.startswith("Q"):
                results.append(qid)

        with self._surface_lock:
            self._surface_cache.setdefault(surface, results)

        self._save_surface_cache()

        return results[:limit]

    async def asearch_wikidata(self, surface: str, limit: int = 1) -> List[str]:
        """
        @brief 异步版本的 Wikidata 搜索接口（基于线程池封装）。
               Async version of Wikidata search API, backed by a thread pool.
        @param surface 表面字符串。Surface form string.
        @param limit 返回的最大实体数量。Maximum number of entity IDs to return.
        @return 匹配到的实体 ID 列表。List of matched entity IDs.
        @note
            - 内部复用同步实现 _search_wikidata，行为保持一致。
              Internally reuse the sync implementation _search_wikidata, keeping behavior identical.
        """
        loop = asyncio.get_running_loop()
        executor = self._get_executor()
        return await loop.run_in_executor(
            executor, self._search_wikidata, surface, limit
        )

    # ------------------------------------------------------------
    # 实体链接：从 tokens 到 QID 列表
    # ------------------------------------------------------------

    def link_entities_from_tokens(self, tokens: Sequence[str]) -> List[str]:
        """
        @brief 基于分词结果执行实体链接，将 token 序列映射为 Wikidata 实体 ID。
               Perform entity linking from token sequence to Wikidata entity IDs.
        @param tokens 文本分词后的 token 序列。Token sequence produced by text preprocessing.
        @return 去重后的实体 ID 列表。Deduplicated list of entity IDs.
        @note
            - 当前实现使用简单的逐 token 规则，每个 token 单独查询 Wikidata 搜索 API。
              The current implementation uses a simple per-token rule: each token is searched individually.
        """
        entities: List[str] = []
        seen: set[str] = set()

        for tok in tokens:
            tok = tok.strip()
            # 过滤掉长度 1 的无意义 token，例如标点或单个汉字。
            if len(tok) <= 1:
                continue

            # 可以视需要添加正则过滤（例如只保留字母/数字/汉字）。
            if not re.search(r"[0-9A-Za-z\u4e00-\u9fff]", tok):
                continue

            qids = self._search_wikidata(tok, limit=1)
            for qid in qids:
                if qid not in seen:
                    seen.add(qid)
                    entities.append(qid)

        logger.info(
            "link_entities_from_tokens: %d tokens -> %d entities",
            len(tokens),
            len(entities),
        )
        return entities

    async def alink_entities_from_tokens(self, tokens: Sequence[str]) -> List[str]:
        """
        @brief 异步版本：基于分词结果的实体链接接口。
               Async version of entity linking from token sequence.
        @param tokens 文本分词后的 token 序列。Token sequence obtained from text preprocessing.
        @return 去重后的实体 ID 列表。Deduplicated list of entity IDs.
        @note
            - 内部仍然调用同步 link_entities_from_tokens，通过线程池 offload。
              Internally calls sync link_entities_from_tokens via the thread pool.
        """
        loop = asyncio.get_running_loop()
        executor = self._get_executor()
        return await loop.run_in_executor(
            executor, self.link_entities_from_tokens, list(tokens)
        )

    # ------------------------------------------------------------
    # 实体链接：从原始文本到 QID 列表（保留接口）
    # ------------------------------------------------------------

    def link_entities(self, text: str) -> List[str]:
        """
        @brief 从原始文本执行实体链接（简单分词 + 实体链接），为兼容保留。
               Entity linking from raw text (simple tokenization + entity linking), kept for compatibility.
        @param text 原始文本字符串。Raw text string.
        @return 实体 ID 列表。List of entity IDs.
        @note
            - 在新版流水线中建议先由上游预处理模块完成分词，再调用 link_entities_from_tokens。
              In the new pipeline, it is recommended that upstream preprocessing handles tokenization explicitly
              and then calls link_entities_from_tokens.
        """
        # 极简 whitespace 分词，只做兜底；正式流程不应依赖该逻辑。
        tokens = re.findall(r"\S+", text)
        return self.link_entities_from_tokens(tokens)

    # ------------------------------------------------------------
    # 邻居查询（同步 & 异步）
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
        # 内存缓存优先（加锁）。
        # Prefer in-memory cache (with lock).
        with self._neighbor_lock:
            cached = self._neighbor_cache.get(entity_id)
        if cached is not None:
            return cached

        # 磁盘缓存其次。
        # Then consult disk cache.
        disk_cached = self._load_neighbors_from_disk(entity_id)
        if disk_cached is not None:
            with self._neighbor_lock:
                self._neighbor_cache[entity_id] = disk_cached
            return disk_cached

        # 发送 SPARQL 查询（当前线程复用自己的 SPARQLWrapper 实例，以支持多线程且复用连接）。
        # Send SPARQL query (reuse the per-thread SPARQLWrapper instance for multi-threading & connection reuse).
        sparql = self._get_sparql()
        query = self._build_neighbor_query(entity_id)
        sparql.setQuery(query)

        try:
            results = sparql.query().convert()
        except Exception as e:  # noqa: BLE001
            logger.error("SPARQL query failed for %s: %s", entity_id, e)
            with self._neighbor_lock:
                self._neighbor_cache[entity_id] = []
            return []

        neighbors: List[str] = []
        for row in results.get("results", {}).get("bindings", []):
            uri = row.get("neighbor", {}).get("value")
            if not uri:
                continue
            qid = str(uri).rsplit("/", 1)[-1]
            if qid and qid != entity_id:
                neighbors.append(qid)

        neighbors = sorted(set(neighbors))

        # 回写内存缓存 & 磁盘缓存。
        # Write back to in-memory and disk caches.
        with self._neighbor_lock:
            self._neighbor_cache[entity_id] = neighbors
        self._save_neighbors_to_disk(entity_id, neighbors)

        logger.info("get_neighbors: %s -> %d neighbors", entity_id, len(neighbors))
        return neighbors

    async def aget_neighbors(self, entity_id: str) -> List[str]:
        """
        @brief 异步获取单个实体的一跳邻居。
               Async version of get_neighbors for a single entity.
        @param entity_id 实体 ID。Entity ID.
        @return 邻居实体 ID 列表。List of neighbor entity IDs.
        """
        loop = asyncio.get_running_loop()
        executor = self._get_executor()
        return await loop.run_in_executor(executor, self.get_neighbors, entity_id)

    def get_entity_contexts(self, entities: Sequence[str]) -> List[List[str]]:
        """
        @brief 获取一组实体各自的一跳邻居列表（实体上下文）。
               Get 1-hop neighbors (entity contexts) for a list of entities.
        @param entities Wikidata 实体 ID 序列。Sequence of Wikidata entity IDs.
        @return 每个实体对应的邻居 ID 列表，顺序与输入对齐。
                List of neighbor ID lists, aligned with input order.
        """
        if not entities:
            return []

        # 1) 对实体列表做“稳定去重”，保留第一次出现的位置，构造映射：
        #    - unique_entities: 去重后的实体列表
        #    - index_map[i] = 原始位置 i 对应在 unique_entities 里的索引
        unique_entities: List[str] = []
        index_map: List[int] = []
        seen: Dict[str, int] = {}

        for eid in entities:
            if eid in seen:
                idx = seen[eid]
            else:
                idx = len(unique_entities)
                unique_entities.append(eid)
                seen[eid] = idx
            index_map.append(idx)

        # 2) 对去重后的实体列表并发调用 get_neighbors
        executor = self._get_executor()
        futures: List[Future[List[str]]] = []
        for eid in unique_entities:
            futures.append(executor.submit(self.get_neighbors, eid))

        unique_contexts: List[List[str]] = []
        for eid, fut in zip(unique_entities, futures):
            try:
                neighbors = fut.result()
            except Exception as e:  # noqa: BLE001
                logger.error("get_entity_contexts: failed for %r: %s", eid, e)
                neighbors = []
            unique_contexts.append(neighbors)

        # 3) 按原始顺序还原：对于原来的第 i 个实体，取 unique_contexts[index_map[i]]
        contexts: List[List[str]] = []
        for idx in index_map:
            contexts.append(unique_contexts[idx])

        return contexts

    async def aget_entity_contexts(self, entities: Sequence[str]) -> List[List[str]]:
        """
        @brief 异步批量获取实体上下文（多实体邻居列表）。
               Async version of get_entity_contexts for a batch of entities.
        @param entities 实体 ID 序列。Sequence of entity IDs.
        @return 邻居实体 ID 列表序列。Sequence of neighbor ID lists.
        """
        if not entities:
            return []

        loop = asyncio.get_running_loop()
        executor = self._get_executor()

        tasks = [
            loop.run_in_executor(executor, self.get_neighbors, eid) for eid in entities
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        contexts: List[List[str]] = []
        for eid, res in zip(entities, results):
            if isinstance(res, Exception):
                logger.error("aget_entity_contexts: failed for %r: %s", eid, res)
                contexts.append([])
            else:
                contexts.append(res)
        return contexts
