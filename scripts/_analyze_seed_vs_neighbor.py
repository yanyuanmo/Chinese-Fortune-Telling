"""分析 graph 管道最终选出的文档中，seed vs neighbor 占比。
验证到底是 graph 检索不准还是 reranker 不准。"""
import json, os

# Load v3 (graph, no HyDE) and v5 (HyDE+graph) detail files  
v3_files = sorted(f for f in os.listdir('benchmarks/results/v8') if f.startswith('graph_rag_v3_cot_only') and f.endswith('_detail.json'))
v5_files = sorted(f for f in os.listdir('benchmarks/results/v8') if f.startswith('graph_rag_v5_hyde_cot') and f.endswith('_detail.json'))

hyde_files = []
for d in ['v5','v5b','v5d','v5e']:
    p = f'benchmarks/results/{d}'
    if os.path.isdir(p):
        for f in os.listdir(p):
            if 'hyde_rerank_bge' in f and f.endswith('_detail.json'):
                hyde_files.append(os.path.join(p, f))
hyde_files.sort()

with open(f'benchmarks/results/v8/{v3_files[-1]}', encoding='utf-8') as fp:
    v3 = json.load(fp)
with open(f'benchmarks/results/v8/{v5_files[-1]}', encoding='utf-8') as fp:
    v5 = json.load(fp)
with open(hyde_files[-1], encoding='utf-8') as fp:
    hyde = json.load(fp)

# 但 detail.json 的 records 只存了 retrieved_contexts (text list)
# 没有 metadata... 我们需要通过运行 retriever 来获取 metadata

# 换一个思路：我们重新跑一次 graph retriever，只跑检索不跑生成，
# 记录 seed_docs 和 neighbor_docs 以及 rerank 后哪些被选入

print("=" * 70)
print("重新运行 Graph Retriever 分析 seed/neighbor 占比")
print("=" * 70)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "api"))

from dotenv import load_dotenv
load_dotenv()

# 加载 vectorstore
os.environ["CHROMA_DIR"] = "./chroma_db_bge"
os.environ["EMBEDDING_MODEL"] = "BAAI/bge-small-zh-v1.5"

import chroma_utils as _cu
_cu._vectorstore = None
_cu._embedding_function = None
from chroma_utils import get_vectorstore

from graph_retriever import GraphRetriever, load_graph_and_index

vs = get_vectorstore()
G, chunk_index = load_graph_and_index("./data/knowledge_graph.pkl", "./data/chunk_index.json")

# Build retriever (v3 = no HyDE)
gr_v3 = GraphRetriever(
    vectorstore=vs, graph=G, chunk_index=chunk_index,
    k=10, hop=1, top_n=7, max_neighbors=30,
    reranker_model="BAAI/bge-reranker-base",
)
print("预加载 BGE reranker...")
gr_v3._get_reranker()
print("就绪 ✓\n")

# Load questions
with open('benchmarks/qa_dataset.json', encoding='utf-8') as fp:
    dataset = json.load(fp)

# Monkey-patch _get_relevant_documents to capture seed/neighbor breakdown
import torch
from langchain_core.documents import Document

def analyze_retrieval(retriever, question):
    """Run retrieval and return detailed breakdown."""
    # Step 1: vector search
    seed_docs = retriever.vectorstore.similarity_search(question, k=retriever.k)
    
    # Get seed chunk ids
    seed_ids = []
    for doc in seed_docs:
        cid = (doc.metadata or {}).get("id") or (doc.metadata or {}).get("chunk_id")
        if cid is None:
            from graph_retriever import _text_to_chunk_id
            cid = _text_to_chunk_id(doc.page_content, retriever.chunk_index)
        seed_ids.append(cid)
    
    # Step 2: graph BFS
    neighbor_ids = []
    seen_ids = set(i for i in seed_ids if i)
    for sid in seed_ids:
        if sid is None:
            continue
        for nbr in retriever._bfs_neighbors(sid):
            if nbr not in seen_ids:
                seen_ids.add(nbr)
                neighbor_ids.append(nbr)
    
    neighbor_docs = []
    for nbr_id in neighbor_ids:
        info = retriever.chunk_index.get(nbr_id)
        if info:
            neighbor_docs.append(Document(
                page_content=info["text"],
                metadata={"book": info["book"], "source": "graph_neighbor"}
            ))
    
    all_docs = seed_docs + neighbor_docs
    n_seed = len(seed_docs)
    n_neighbor = len(neighbor_docs)
    n_total = len(all_docs)
    
    # Step 4: rerank
    reranker = retriever._get_reranker()
    tok, mdl = reranker
    with torch.no_grad():
        inputs = tok(
            [question] * n_total,
            [d.page_content for d in all_docs],
            return_tensors="pt", truncation=True, max_length=256, padding=True,
        )
        scores = mdl(**inputs).logits.squeeze(-1).tolist()
        if isinstance(scores, float):
            scores = [scores]
    
    # Tag each doc
    tagged = []
    for idx, (score, doc) in enumerate(zip(scores, all_docs)):
        is_seed = idx < n_seed
        tagged.append({
            "score": score,
            "is_seed": is_seed,
            "book": (doc.metadata or {}).get("book", "?"),
            "text_head": doc.page_content[:60],
        })
    
    tagged.sort(key=lambda x: x["score"], reverse=True)
    
    # Count in top_n
    top7 = tagged[:7]
    seeds_in_top7 = sum(1 for t in top7 if t["is_seed"])
    neighbors_in_top7 = sum(1 for t in top7 if not t["is_seed"])
    
    return {
        "n_seed": n_seed,
        "n_neighbor": n_neighbor,
        "n_total": n_total,
        "seeds_in_top7": seeds_in_top7,
        "neighbors_in_top7": neighbors_in_top7,
        "top7_scores": [round(t["score"], 3) for t in top7],
        "top7_sources": ["S" if t["is_seed"] else "N" for t in top7],
        "seed_scores": sorted([t["score"] for t in tagged if t["is_seed"]], reverse=True),
        "neighbor_scores": sorted([t["score"] for t in tagged if not t["is_seed"]], reverse=True),
    }

print(f"{'Q':>2}  {'seeds':>5} {'nbrs':>5} {'total':>5}  {'S_in7':>5} {'N_in7':>5}  {'top7组成':>12}  {'seed均分':>8} {'nbr均分':>8}")
print("-" * 80)

total_seeds_in_7 = 0
total_neighbors_in_7 = 0
all_seed_scores = []
all_nbr_scores = []

for i, item in enumerate(dataset):
    q = item["question"]
    result = analyze_retrieval(gr_v3, q)
    total_seeds_in_7 += result["seeds_in_top7"]
    total_neighbors_in_7 += result["neighbors_in_top7"]
    all_seed_scores.extend(result["seed_scores"])
    all_nbr_scores.extend(result["neighbor_scores"])
    
    src_str = "".join(result["top7_sources"])
    seed_mean = sum(result["seed_scores"]) / max(len(result["seed_scores"]), 1)
    nbr_mean = sum(result["neighbor_scores"]) / max(len(result["neighbor_scores"]), 1) if result["neighbor_scores"] else 0
    
    print(f"{i+1:>2}   {result['n_seed']:>4}  {result['n_neighbor']:>4}  {result['n_total']:>4}    {result['seeds_in_top7']:>3}    {result['neighbors_in_top7']:>3}   {src_str:>10}   {seed_mean:>7.2f}  {nbr_mean:>7.2f}")

n = len(dataset)
print("-" * 80)
print(f"\n汇总 ({n} questions):")
print(f"  top7 中 seed 总数:     {total_seeds_in_7}/{n*7} ({total_seeds_in_7/(n*7):.1%})")
print(f"  top7 中 neighbor 总数: {total_neighbors_in_7}/{n*7} ({total_neighbors_in_7/(n*7):.1%})")

s_mean = sum(all_seed_scores) / len(all_seed_scores) if all_seed_scores else 0
n_mean = sum(all_nbr_scores) / len(all_nbr_scores) if all_nbr_scores else 0
print(f"\n  所有 seed 平均 reranker 分: {s_mean:.3f}")
print(f"  所有 neighbor 平均 reranker 分: {n_mean:.3f}")
print(f"  差值 (seed - neighbor): {s_mean - n_mean:+.3f}")

print(f"""
\n解读:
  如果 top7 大部分是 seed → reranker 能有效过滤 graph 噪声，那 graph 提供的邻居就是拉了后腿
  如果 top7 有大量 neighbor → 说明 neighbor 的 reranker 分和 seed 差不多，reranker 无法区分
""")
