"""build_knowledge_graph.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
从 chroma_db_bge 中提取所有 chunk，识别命理桥接术语，
构建跨书籍知识图谱（NetworkX）。

图结构
────────────────────────────────────────────────────────
节点：每个 chunk（节点 ID = chroma 内部 id）
      属性：book, text, bridge_terms
边：  来自不同书籍、共享≥min_weight 条桥接术语的 chunk 对
      属性：shared_terms（共享术语列表）、weight（数量）

输出（到 --output-dir，默认 ./data/）
────────────────────────────────────────────────────────
  knowledge_graph.pkl          NetworkX Graph（pickle）
  chunk_index.json             {id: {book, text, bridge_terms}}
  graph_stats.json             节点数/边数/每书籍分布/桥接术语频率

用法
────────────────────────────────────────────────────────
  cd E:\\repos\\Chinese-Fortune-Telling
  python scripts/build_knowledge_graph.py
  python scripts/build_knowledge_graph.py --chroma-dir ./chroma_db_bge --min-weight 1
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ── 路径 ──────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "api"))
from dotenv import load_dotenv
load_dotenv(override=True)

# ── 命理桥接术语（与 generate_qa_multihop.py 保持一致）────────────────────────
BRIDGE_TERMS: list[str] = [
    "官星", "财星", "印绶", "伤官", "格局", "用神", "日主", "五行",
    "阴阳", "天干", "地支", "命宫", "大运", "流年", "正官", "偏官",
    "正财", "偏财", "正印", "偏印", "比肩", "劫财", "食神", "七煞",
    "羊刃", "日禄", "月令", "扶抑", "调候", "通关", "专旺", "从格",
    "化气", "合化", "刑冲", "日元",
]


def extract_bridge_terms(text: str) -> list[str]:
    """返回文本中出现的桥接术语列表（可重复，按出现顺序）。"""
    found = []
    for term in BRIDGE_TERMS:
        if term in text:
            found.append(term)
    return found


def build_graph(
    chroma_dir: str,
    embedding_model: str,
    min_weight: int,
    min_chunk_len: int,
    max_degree: int = 0,
) -> tuple:
    """
    加载 chroma_db_bge → 提取桥接术语 → 构建 NetworkX 图。
    返回 (graph, chunk_index)。
    """
    import networkx as nx

    # ── 加载 vectorstore ────────────────────────────────────────────────────
    os.environ["CHROMA_DIR"] = chroma_dir
    os.environ["EMBEDDING_MODEL"] = embedding_model

    import chroma_utils as _cu
    _cu._vectorstore = None
    _cu._embedding_function = None
    from chroma_utils import get_vectorstore

    print("Loading vectorstore …", flush=True)
    vs = get_vectorstore()
    result = vs._collection.get(include=["documents", "metadatas"])
    ids       = result["ids"]
    docs      = result["documents"]
    metas     = result["metadatas"]
    print(f"  Total chunks: {len(ids)}", flush=True)

    # ── 过滤短 chunk，提取术语 ────────────────────────────────────────────────
    chunk_index: dict[str, dict] = {}
    by_book: dict[str, list[str]] = defaultdict(list)

    for cid, text, meta in zip(ids, docs, metas):
        if not text or len(text) < min_chunk_len:
            continue
        book = (meta or {}).get("book", "unknown")
        terms = extract_bridge_terms(text)
        chunk_index[cid] = {
            "book":         book,
            "text":         text,
            "bridge_terms": terms,
        }
        by_book[book].append(cid)

    print(f"  Chunks after min_len={min_chunk_len} filter: {len(chunk_index)}", flush=True)
    for bk, cids in sorted(by_book.items()):
        print(f"    {bk}: {len(cids)} chunks", flush=True)

    # ── 构建图 ──────────────────────────────────────────────────────────────
    print("Building knowledge graph …", flush=True)
    import math
    G = nx.Graph()

    # 添加节点
    for cid, info in chunk_index.items():
        G.add_node(cid, book=info["book"], bridge_terms=info["bridge_terms"])

    # 按术语建倒排索引（只取跨书配对）
    term_to_chunks: dict[str, list[str]] = defaultdict(list)
    for cid, info in chunk_index.items():
        for term in set(info["bridge_terms"]):   # set 去重，避免同一术语多次出现
            term_to_chunks[term].append(cid)

    # ── IDF 权重：稀有术语贡献大，高频术语贡献小 ──────────────────────────
    N = len(chunk_index)
    term_idf: dict[str, float] = {}
    for term, cids_list in term_to_chunks.items():
        df = len(cids_list)
        term_idf[term] = math.log(N / df) if df > 0 else 0.0

    print(f"  IDF weights (sample): ", flush=True)
    for term in sorted(term_idf, key=term_idf.get, reverse=True)[:5]:
        print(f"    {term}: IDF={term_idf[term]:.3f} (df={len(term_to_chunks[term])})", flush=True)
    for term in sorted(term_idf, key=term_idf.get)[:3]:
        print(f"    {term}: IDF={term_idf[term]:.3f} (df={len(term_to_chunks[term])})", flush=True)

    # 枚举跨书边（IDF 加权）
    edge_weight: dict[tuple, float] = defaultdict(float)
    edge_terms:  dict[tuple, list]  = defaultdict(list)

    for term, cids_list in term_to_chunks.items():
        idf = term_idf[term]
        # 只处理当前术语涉及的 chunk 对
        for i in range(len(cids_list)):
            for j in range(i + 1, len(cids_list)):
                a, b = cids_list[i], cids_list[j]
                if chunk_index[a]["book"] == chunk_index[b]["book"]:
                    continue  # 同书不建边
                key = (min(a, b), max(a, b))
                edge_weight[key] += idf
                edge_terms[key].append(term)

    # 写入满足 min_weight 的边
    edges_added = 0
    for (a, b), w in edge_weight.items():
        if w >= min_weight:
            G.add_edge(a, b, weight=round(w, 4), shared_terms=edge_terms[(a, b)])
            edges_added += 1

    print(f"  Edges before pruning: {edges_added}  (min_weight={min_weight})", flush=True)

    # ── 度限制剪枝：每个节点只保留权重最高的 top-K 条边 ─────────────────────
    # 策略：保留边 (A,B) 如果 A 或 B 其中任一方将其排在自己的 top-K 内
    if max_degree > 0:
        edges_before = G.number_of_edges()
        edges_to_keep = set()
        for node in G.nodes():
            neighbors = list(G[node].items())  # [(nbr, edge_data), ...]
            neighbors.sort(key=lambda x: x[1].get("weight", 0), reverse=True)
            for nbr, _ in neighbors[:max_degree]:
                edges_to_keep.add((min(node, nbr), max(node, nbr)))
        edges_to_remove = []
        for u, v in G.edges():
            if (min(u, v), max(u, v)) not in edges_to_keep:
                edges_to_remove.append((u, v))
        G.remove_edges_from(edges_to_remove)
        print(f"  Degree pruning (max_degree={max_degree}): "
              f"{edges_before} → {G.number_of_edges()} edges "
              f"(removed {len(edges_to_remove)})", flush=True)

    avg_degree = (sum(dict(G.degree()).values()) / G.number_of_nodes()
                  if G.number_of_nodes() else 0)
    print(f"  Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}  "
          f"avg_degree: {avg_degree:.1f}", flush=True)

    return G, chunk_index


def save_outputs(
    G,
    chunk_index: dict,
    output_dir: Path,
) -> None:
    import networkx as nx

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── pickle 图 ─────────────────────────────────────────────────────────
    graph_pkl = output_dir / "knowledge_graph.pkl"
    with open(graph_pkl, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Graph → {graph_pkl}", flush=True)

    # ── chunk 索引（JSON，不含全文 → 太大；只含 book + bridge_terms + 200 字预览）──
    # 注意：graph_retriever 需要全文，所以这里保留全文
    chunk_idx_path = output_dir / "chunk_index.json"
    with open(chunk_idx_path, "w", encoding="utf-8") as f:
        json.dump(chunk_index, f, ensure_ascii=False, indent=2)
    print(f"  Chunk index → {chunk_idx_path}", flush=True)

    # ── 统计 ─────────────────────────────────────────────────────────────
    from collections import Counter
    book_counts: Counter = Counter()
    term_counts: Counter = Counter()
    for _, info in chunk_index.items():
        book_counts[info["book"]] += 1
        for t in info["bridge_terms"]:
            term_counts[t] += 1

    edge_weights = [data["weight"] for _, _, data in G.edges(data=True)]
    avg_degree = (sum(dict(G.degree()).values()) / G.number_of_nodes()
                  if G.number_of_nodes() else 0)

    stats = {
        "nodes":           G.number_of_nodes(),
        "edges":           G.number_of_edges(),
        "avg_degree":      round(avg_degree, 2),
        "by_book":         dict(book_counts),
        "bridge_term_freq": dict(term_counts.most_common(20)),
        "edge_weight_dist": {
            "min":    min(edge_weights) if edge_weights else 0,
            "max":    max(edge_weights) if edge_weights else 0,
            "mean":   round(sum(edge_weights) / len(edge_weights), 2) if edge_weights else 0,
        },
    }
    stats_path = output_dir / "graph_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  Stats → {stats_path}", flush=True)
    print(f"\n  Top bridge terms: {list(term_counts.most_common(10))}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Build cross-book knowledge graph from chroma_db_bge")
    parser.add_argument("--chroma-dir", default="./chroma_db_bge",
                        help="Path to chroma vector store (default: ./chroma_db_bge)")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-zh-v1.5",
                        help="Embedding model name")
    parser.add_argument("--min-weight", type=int, default=1,
                        help="Min shared bridge terms to create an edge (default: 1)")
    parser.add_argument("--min-chunk-len", type=int, default=50,
                        help="Minimum chunk character length to include (default: 50)")
    parser.add_argument("--max-degree", type=int, default=0,
                        help="Max edges per node (0=no limit). Keeps top-K by weight. (default: 0)")
    parser.add_argument("--output-dir", default="./data",
                        help="Directory to save graph files (default: ./data)")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    os.chdir(root)

    print("=" * 60, flush=True)
    print("  Building Knowledge Graph for Graph RAG", flush=True)
    print("=" * 60, flush=True)
    print(f"  chroma_dir:     {args.chroma_dir}", flush=True)
    print(f"  embedding:      {args.embedding_model}", flush=True)
    print(f"  min_weight:     {args.min_weight}", flush=True)
    print(f"  min_chunk_len:  {args.min_chunk_len}", flush=True)
    print(f"  max_degree:     {args.max_degree}", flush=True)
    print(f"  output_dir:     {args.output_dir}", flush=True)
    print(flush=True)

    G, chunk_index = build_graph(
        chroma_dir     = args.chroma_dir,
        embedding_model= args.embedding_model,
        min_weight     = args.min_weight,
        min_chunk_len  = args.min_chunk_len,
        max_degree     = args.max_degree,
    )

    save_outputs(G, chunk_index, Path(args.output_dir))

    print("\n✅  Knowledge graph built successfully.", flush=True)


if __name__ == "__main__":
    main()
