"""_analyze_vector_filter.py
比较 v3 CoT (无向量过滤) vs v7 vf50 (vector_filter_k=50) 的检索层差异。
不需要 LLM API，仅用本地模型。

输出：
- 候选池大小对比
- reranker 分数分布对比
- top-7 中 seed/neighbor 比例对比
- 具体检索文档差异
"""

import json, os, sys, yaml
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "api"))
sys.path.insert(0, str(ROOT / "scripts"))

os.environ["CHROMA_DIR"]      = str(ROOT / "chroma_db_bge")
os.environ["EMBEDDING_MODEL"] = "BAAI/bge-small-zh-v1.5"

from chroma_utils import get_vectorstore
import rag_bench as rb
import torch

# ── 加载两个 retriever ────────────────────────────────────────────────────────
print("Loading retrievers...", flush=True)
vs = get_vectorstore()

cfg_v3  = yaml.safe_load(open(ROOT / "configs/rag/v8/graph_rag_v3_cot_only.yaml", encoding="utf-8"))
cfg_vf50 = yaml.safe_load(open(ROOT / "configs/rag/v8/graph_rag_v7_vf50.yaml", encoding="utf-8"))

ret_v3  = rb.build_retriever(cfg_v3, vs)
ret_vf50 = rb.build_retriever(cfg_vf50, vs)
print(f"v3:   vector_filter_k={ret_v3.vector_filter_k}, max_neighbors={ret_v3.max_neighbors}")
print(f"vf50: vector_filter_k={ret_vf50.vector_filter_k}, max_neighbors={ret_vf50.max_neighbors}")

# ── 加载 multihop 数据集 ──────────────────────────────────────────────────────
with open(ROOT / "benchmarks/qa_multihop.json", encoding="utf-8") as f:
    qa = json.load(f)
questions = [q["question"] for q in qa]

# ── 修改 retriever 以捕获中间状态 ──────────────────────────────────────────────
# Monkey-patch to capture candidate pool size and scores
def patched_get_relevant(self, query, *, run_manager=None):
    """Instrumented version that captures internal stats."""
    from langchain_core.documents import Document

    # Step 1: seeds
    search_text = query
    if self.hyde_llm is not None:
        try:
            search_text = self.hyde_llm.invoke(
                f"你是中国传统命理学专家。请根据以下问题，仿照古典命理文献的风格，写一段 80-150 字的原文片段。\n问题：{query}"
            ).content.strip()
        except:
            search_text = query

    try:
        seed_docs = self.vectorstore.similarity_search(search_text, k=self.k)
    except:
        seed_docs = []

    seed_ids = []
    for doc in seed_docs:
        cid = (doc.metadata or {}).get("id") or (doc.metadata or {}).get("chunk_id")
        if cid is None:
            from graph_retriever import _text_to_chunk_id
            cid = _text_to_chunk_id(doc.page_content, self.chunk_index)
        seed_ids.append(cid)

    # Step 1b: vector whitelist
    vector_whitelist = None
    if self.vector_filter_k > 0:
        try:
            wide_docs = self.vectorstore.similarity_search(query, k=self.vector_filter_k)
            wl = set()
            for wd in wide_docs:
                from graph_retriever import _text_to_chunk_id
                wid = (wd.metadata or {}).get("id") or (wd.metadata or {}).get("chunk_id")
                if wid is None:
                    wid = _text_to_chunk_id(wd.page_content, self.chunk_index)
                if wid:
                    wl.add(wid)
            vector_whitelist = wl
        except:
            vector_whitelist = None

    # Step 2: BFS neighbors
    neighbor_ids = []
    seen_ids = set(i for i in seed_ids if i)
    all_bfs_neighbors = []  # before filtering
    for sid in seed_ids:
        if sid is None:
            continue
        for nbr in self._bfs_neighbors(sid):
            if nbr not in seen_ids:
                seen_ids.add(nbr)
                all_bfs_neighbors.append(nbr)
                if vector_whitelist is not None and nbr not in vector_whitelist:
                    continue
                neighbor_ids.append(nbr)

    # Step 3: neighbor docs
    neighbor_docs = []
    for nbr_id in neighbor_ids:
        info = self.chunk_index.get(nbr_id)
        if info:
            neighbor_docs.append(Document(
                page_content=info["text"],
                metadata={"book": info["book"], "source": "graph_neighbor"}
            ))

    all_docs = seed_docs + neighbor_docs

    # Step 4: reranker
    reranker = self._get_reranker()
    scores_list = []
    sources_list = []
    if reranker and len(all_docs) > self.top_n:
        tok, mdl = reranker
        with torch.no_grad():
            inputs = tok(
                [query] * len(all_docs),
                [d.page_content for d in all_docs],
                return_tensors="pt", truncation=True, max_length=256, padding=True,
            )
            scores = mdl(**inputs).logits.squeeze(-1).tolist()
            if isinstance(scores, float):
                scores = [scores]

        for i, (s, doc) in enumerate(zip(scores, all_docs)):
            src = "seed" if i < len(seed_docs) else "neighbor"
            scores_list.append(s)
            sources_list.append(src)

        ranked = sorted(zip(scores, all_docs), key=lambda x: x[0], reverse=True)
        selected = ranked[:self.top_n]
    else:
        selected = [(0, d) for d in all_docs[:self.top_n]]

    # Return stats
    stats = {
        "n_seeds": len(seed_docs),
        "n_bfs_neighbors_raw": len(all_bfs_neighbors),
        "n_neighbors_after_filter": len(neighbor_ids),
        "n_candidates": len(all_docs),
        "filtered_pct": (1 - len(neighbor_ids)/max(len(all_bfs_neighbors),1)) * 100 if all_bfs_neighbors else 0,
        "seed_scores": [s for s, src in zip(scores_list, sources_list) if src == "seed"],
        "neighbor_scores": [s for s, src in zip(scores_list, sources_list) if src == "neighbor"],
        "top7_seeds": sum(1 for _, d in selected if d.metadata.get("source") != "graph_neighbor"),
        "top7_neighbors": sum(1 for _, d in selected if d.metadata.get("source") == "graph_neighbor"),
        "top7_books": len(set((d.metadata or {}).get("book","") for _, d in selected)),
    }
    return [d for _, d in selected], stats

# ── 逐题跑两个 retriever ──────────────────────────────────────────────────────
print(f"\nRunning {len(questions)} questions through both retrievers...\n")

v3_stats_all = []
vf50_stats_all = []

for i, q in enumerate(questions):
    docs_v3, stats_v3 = patched_get_relevant(ret_v3, q)
    docs_vf50, stats_vf50 = patched_get_relevant(ret_vf50, q)
    v3_stats_all.append(stats_v3)
    vf50_stats_all.append(stats_vf50)

    # Show per-question summary
    filtered = stats_vf50["filtered_pct"]
    print(f"Q{i+1:>2}: v3 cand={stats_v3['n_candidates']:>3}  "
          f"vf50 cand={stats_vf50['n_candidates']:>3} "
          f"(filtered {filtered:.0f}%)  "
          f"top7: v3={stats_v3['top7_seeds']}s/{stats_v3['top7_neighbors']}n  "
          f"vf50={stats_vf50['top7_seeds']}s/{stats_vf50['top7_neighbors']}n")

# ── 汇总 ───────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("AGGREGATE COMPARISON")
print("=" * 70)

for name, stats_all in [("v3 (no filter)", v3_stats_all), ("vf50 (top-50 filter)", vf50_stats_all)]:
    n = len(stats_all)
    avg_cand = sum(s["n_candidates"] for s in stats_all) / n
    avg_raw = sum(s["n_bfs_neighbors_raw"] for s in stats_all) / n
    avg_kept = sum(s["n_neighbors_after_filter"] for s in stats_all) / n
    avg_filtered = sum(s["filtered_pct"] for s in stats_all) / n
    avg_seed_top7 = sum(s["top7_seeds"] for s in stats_all) / n
    avg_nbr_top7 = sum(s["top7_neighbors"] for s in stats_all) / n
    avg_books = sum(s["top7_books"] for s in stats_all) / n

    all_seed_scores = [s for st in stats_all for s in st["seed_scores"]]
    all_nbr_scores = [s for st in stats_all for s in st["neighbor_scores"]]
    avg_seed_score = sum(all_seed_scores) / max(len(all_seed_scores), 1)
    avg_nbr_score = sum(all_nbr_scores) / max(len(all_nbr_scores), 1)

    print(f"\n{name}:")
    print(f"  avg BFS neighbors (raw): {avg_raw:.1f}")
    print(f"  avg neighbors kept:      {avg_kept:.1f}")
    print(f"  avg filtered out:        {avg_filtered:.1f}%")
    print(f"  avg candidates:          {avg_cand:.1f}")
    print(f"  avg top-7 seed/neighbor: {avg_seed_top7:.1f} / {avg_nbr_top7:.1f}")
    print(f"  avg top-7 books:         {avg_books:.1f}")
    print(f"  avg reranker score (seed):     {avg_seed_score:.3f}")
    print(f"  avg reranker score (neighbor): {avg_nbr_score:.3f}")

print("\nDone.")
