"""graph_retriever.py
~~~~~~~~~~~~~~~~~~~~
Graph-aware LangChain retriever，用于 Graph RAG。

检索步骤
────────────────────────────────────────────────────────
1. 向量宽召回（dense seed）
   用 vectorstore.similarity_search(query, k=k) 获取 seed 文档。

2. 知识图谱邻居扩展
   对每个 seed chunk，在 knowledge_graph 中做 hop 跳 BFS，
   找到来自其他书籍的邻居 chunk。

3. 合并去重
   将 seed + neighbors 合并，去重。

4. BGE 交叉编码器精排（可选）
   用原始问题对全量候选精排（不截断），CrossEncoder 分数即排名依据，
   保留 top_n 个，并做书籍多样性修正。

接口
────────────────────────────────────────────────────────
  from api.graph_retriever import GraphRetriever
  retriever = GraphRetriever(
      vectorstore = vs,
      graph       = G,
      chunk_index = chunk_index,   # {id: {book, text, bridge_terms}}
      k           = 10,
      hop         = 1,
      top_n       = 7,
      reranker_model = "BAAI/bge-reranker-base",
  )
  docs = retriever.invoke("如何判断正财格")

配置兼容
────────────────────────────────────────────────────────
  与 rag_bench.py 中的 build_retriever() 通过
  retrieval.type = "graph_rag" 调用。
"""

from __future__ import annotations

from typing import Any, List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class GraphRetriever(BaseRetriever):
    """
    Graph-aware retriever。

    Parameters
    ----------
    vectorstore : Chroma
        chroma_db_bge 实例（已初始化）。
    graph : networkx.Graph
        由 build_knowledge_graph.py 生成的知识图谱。
    chunk_index : dict
        {chunk_id: {book, text, bridge_terms}}，由 chunk_index.json 加载。
    k : int
        向量宽召回数量（seed 数量）。
    hop : int
        图遍历跳数（1 = 只取直接邻居）。
    top_n : int
        最终返回文档数量（reranker 输出）。
    reranker_model : str
        BGE 交叉编码器模型名称。空字符串则跳过重排。
    max_neighbors : int
        每个 seed 最多扩展多少邻居（防止高 degree 节点爆炸）。
    """

    # ── Pydantic 字段（BaseRetriever 用 Pydantic v1/v2 做序列化）─────────────
    vectorstore: Any
    graph:       Any
    chunk_index: dict
    k:           int   = 10
    hop:         int   = 1
    top_n:       int   = 7
    reranker_model: str = "BAAI/bge-reranker-base"
    max_neighbors:  int = 30
    vector_filter_k: int = 0       # >0 时：用 top-K 向量结果做邻居白名单
    hyde_llm:    Any  = None       # 可选：传入 ChatOpenAI 实例以启用 HyDE 种子

    # 懒加载 reranker（避免 Pydantic 序列化 HuggingFace 对象）
    _reranker: Any = None

    class Config:
        arbitrary_types_allowed = True

    # ── 内部工具 ─────────────────────────────────────────────────────────────

    def _get_reranker(self):
        """Lazily load BGE cross-encoder. Uses XLMRobertaForSequenceClassification
        directly (bge-reranker-base IS xlm-roberta) to avoid the slow
        importlib.metadata entry-point scan that AutoModelForSequenceClassification
        triggers on Windows with large conda environments."""
        if self._reranker is None and self.reranker_model:
            from transformers import XLMRobertaTokenizerFast, XLMRobertaForSequenceClassification
            import torch
            tok = XLMRobertaTokenizerFast.from_pretrained(self.reranker_model)
            mdl = XLMRobertaForSequenceClassification.from_pretrained(self.reranker_model)
            mdl.eval()
            object.__setattr__(self, "_reranker", (tok, mdl))
        return self._reranker

    def _bfs_neighbors(self, seed_id: str) -> list[str]:
        """从 seed_id 出发做加权 BFS，优先沿高权重边扩展。
        返回最多 max_neighbors 个异书邻居 id。"""
        if seed_id not in self.graph:
            return []
        seed_book = self.chunk_index.get(seed_id, {}).get("book", "")
        visited = {seed_id}
        frontier = [seed_id]
        neighbors: list[str] = []

        for _ in range(self.hop):
            next_frontier: list[str] = []
            for node in frontier:
                # 按边权重降序排列邻居，优先探索强关联节点
                adj = [
                    (nbr, self.graph[node][nbr].get("weight", 0))
                    for nbr in self.graph.neighbors(node)
                    if nbr not in visited
                ]
                adj.sort(key=lambda x: x[1], reverse=True)

                for nbr, _w in adj:
                    if nbr in visited:
                        continue
                    visited.add(nbr)
                    nbr_book = self.chunk_index.get(nbr, {}).get("book", "")
                    if nbr_book != seed_book:           # 只取异书邻居
                        neighbors.append(nbr)
                        if len(neighbors) >= self.max_neighbors:
                            return neighbors
                    next_frontier.append(nbr)
            frontier = next_frontier

        return neighbors

    # ── LangChain 接口 ────────────────────────────────────────────────────────

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> List[Document]:

        # ─ Step 1: 向量宽召回 seed docs ─────────────────────────────────────
        # 如果配置了 HyDE LLM，先生成仿古文片段用于向量检索（种子更精准）
        search_text = query
        if self.hyde_llm is not None:
            _HYDE_PROMPT = (
                "你是中国传统命理学专家。请根据以下问题，"
                "仿照古典命理文献（文言文/半文言文）的风格，"
                "写一段 80-150 字的原文片段，直接包含问题答案所涉及的术语和论述。"
                "只输出片段本身，不要标题、序号或解释。\n\n"
                "问题：{question}"
            )
            try:
                search_text = self.hyde_llm.invoke(
                    _HYDE_PROMPT.format(question=query)
                ).content.strip()
            except Exception:
                search_text = query  # 生成失败退回原始问题

        try:
            seed_docs: list[Document] = self.vectorstore.similarity_search(search_text, k=self.k)
        except Exception:
            seed_docs = []

        # 获取 seed chunk ids（chroma 存在 metadata 或 id 字段）
        seed_ids: list[str] = []
        for doc in seed_docs:
            cid = (doc.metadata or {}).get("id") or (doc.metadata or {}).get("chunk_id")
            if cid is None:
                # 用文本哈希作为 fallback key 做图查找
                cid = _text_to_chunk_id(doc.page_content, self.chunk_index)
            seed_ids.append(cid)

        # ─ Step 1b (可选): 向量白名单 ─────────────────────────────────────
        # 如果设置了 vector_filter_k，做一次宽召回获取语义可接受的 chunk id 集合，
        # 后续 BFS 邻居必须在此白名单中才保留（过滤掉语义上完全无关的拓扑邻居）。
        vector_whitelist: set[str] | None = None
        if self.vector_filter_k > 0:
            try:
                wide_docs = self.vectorstore.similarity_search(query, k=self.vector_filter_k)
                wl: set[str] = set()
                for wd in wide_docs:
                    wid = (wd.metadata or {}).get("id") or (wd.metadata or {}).get("chunk_id")
                    if wid is None:
                        wid = _text_to_chunk_id(wd.page_content, self.chunk_index)
                    if wid:
                        wl.add(wid)
                vector_whitelist = wl
            except Exception:
                vector_whitelist = None  # 失败则不过滤

        # ─ Step 2: 图遍历 → 异书邻居 ids ────────────────────────────────────
        neighbor_ids: list[str] = []
        seen_ids: set[str] = set(i for i in seed_ids if i)
        for sid in seed_ids:
            if sid is None:
                continue
            for nbr in self._bfs_neighbors(sid):
                if nbr not in seen_ids:
                    seen_ids.add(nbr)
                    # 向量白名单过滤：只保留语义上也相关的邻居
                    if vector_whitelist is not None and nbr not in vector_whitelist:
                        continue
                    neighbor_ids.append(nbr)

        # ─ Step 3: 将邻居 ids 转为 Document 对象 ────────────────────────────
        neighbor_docs: list[Document] = []
        for nbr_id in neighbor_ids:
            info = self.chunk_index.get(nbr_id)
            if info:
                neighbor_docs.append(Document(
                    page_content=info["text"],
                    metadata={
                        "book": info["book"],
                        "source": "graph_neighbor",
                        "bridge_terms": ",".join(info.get("bridge_terms", [])),
                    }
                ))

        all_docs = seed_docs + neighbor_docs

        if not all_docs:
            return []

        # ─ Step 4: BGE 精排 + 书籍多样性保证 ────────────────────────────────
        reranker = self._get_reranker()
        if reranker and len(all_docs) > self.top_n:
            import torch
            tok, mdl = reranker
            # 直接对全量候选精排（seeds + neighbors 最多约 40 条），
            # 让 CrossEncoder 的分数决定排序，不再用 interleave 做启发式截断。
            candidates = all_docs
            with torch.no_grad():
                inputs = tok(
                    [query] * len(candidates),
                    [d.page_content for d in candidates],
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                    padding=True,
                )
                scores = mdl(**inputs).logits.squeeze(-1).tolist()
                if isinstance(scores, float):   # single-doc edge case
                    scores = [scores]
            ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

            # 多样性选择：先用纯分数选 top_n，再补足"最少 1 个不同书"的槽位
            selected = _diverse_top_n(ranked, self.top_n)
            return selected

        return all_docs[: self.top_n]


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def _interleave_by_book(docs: list) -> list:
    """按书籍交错排列文档，使每本书都有代表进入候选窗口。
    例如 [A1,A2,A3,B1,B2,C1] → [A1,B1,C1,A2,B2,A3]
    """
    from collections import defaultdict
    buckets: dict[str, list] = defaultdict(list)
    for doc in docs:
        book = (doc.metadata or {}).get("book", "unknown")
        buckets[book].append(doc)
    result = []
    max_len = max(len(v) for v in buckets.values())
    for i in range(max_len):
        for book_docs in buckets.values():
            if i < len(book_docs):
                result.append(book_docs[i])
    return result


def _diverse_top_n(ranked: list[tuple], top_n: int) -> list:
    """从 (score, doc) 有序列表中选 top_n，同时保证最终结果里
    至少包含 2 种不同书籍（如果候选中有的话）。
    策略：先贪心选 top_n，若书籍种类<2 则把最低分的单书 doc 替换为分数最高的异书 doc。
    """
    selected = [doc for _, doc in ranked[:top_n]]
    books_in = {(doc.metadata or {}).get("book", "") for doc in selected}
    if len(books_in) >= 2:
        return selected  # 已经有多样性，直接返回

    # 找候选中分数最高的、不在 selected 中的异书 doc
    selected_set = set(id(d) for d in selected)
    for _, doc in ranked[top_n:]:
        b = (doc.metadata or {}).get("book", "")
        if b not in books_in:
            # 把 selected 中分数最低的同书 doc 换掉
            selected[-1] = doc
            break
    return selected


def _text_to_chunk_id(text: str, chunk_index: dict) -> str | None:
    """通过文本前 80 字符反向查找 chunk_id。用于 chroma 没有存 id 的情况。"""
    prefix = text[:80]
    for cid, info in chunk_index.items():
        if info["text"][:80] == prefix:
            return cid
    return None


def load_graph_and_index(graph_path: str, chunk_index_path: str) -> tuple:
    """加载 knowledge_graph.pkl 和 chunk_index.json，返回 (graph, chunk_index)。"""
    import json
    import pickle

    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    with open(chunk_index_path, encoding="utf-8") as f:
        chunk_index = json.load(f)
    return G, chunk_index
