"""Analyze hop1 benchmark failures to identify improvement opportunities."""
import json, pathlib

base = pathlib.Path("benchmarks/results/multihop/20260416_225409")
hop1 = [json.loads(l) for l in open(base / "checkpoint_graph_rag_v3_idf_hop1.jsonl", "r", encoding="utf-8")]

# Deep dive into chain=0 questions
for qi in [22, 30, 31]:
    h = hop1[qi]
    cs = h["chain_score"]
    print(f"=== Q{qi+1} (chain={cs}) ===")
    print(f"  retrieved_books: {h['retrieved_books']}")
    print(f"  source_books: {h['source_books']}")
    print(f"  step_scores: {h['step_scores']}")
    comment = h.get("comment", "")
    print(f"  comment: {comment[:300]}")
    print(f"  answer[:400]: {h['answer'][:400]}")
    print()
    print(f"  golden[:400]: {h['golden_answer'][:400]}")
    print()
    # Show reasoning chain
    chain = h.get("reasoning_chain", [])
    print(f"  reasoning_chain ({len(chain)} steps):")
    for j, step in enumerate(chain):
        print(f"    [{j}] {str(step)[:100]}")
    print()
    print("=" * 80)
    print()

# Also check: are low-score questions a retrieval problem or generation problem?
print("\n=== 所有题的 retrieved_books 覆盖情况 ===")
for i, h in enumerate(hop1):
    src = set(h.get("source_books", []))
    ret = set(h.get("retrieved_books", []))
    missing = src - ret
    cs = h["chain_score"]
    if missing:
        print(f"  Q{i+1} chain={cs:.3f} MISSING books: {missing}")
    elif cs <= 0.5:
        print(f"  Q{i+1} chain={cs:.3f} books OK but low score -- generation problem?")
