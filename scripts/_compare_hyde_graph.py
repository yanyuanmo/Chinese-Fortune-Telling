"""Compare v3 CoT (Graph only) vs v5 (HyDE+Graph) benchmark results."""
import json

# Load results
v3_lines = open("benchmarks/results/multihop/20260417_092940/checkpoint_graph_rag_v3_cot_only.jsonl", encoding="utf-8").readlines()
v3 = [json.loads(l) for l in v3_lines]

v5_lines = open("benchmarks/results/multihop/20260417_153733/checkpoint_graph_rag_v5_hyde_cot.jsonl", encoding="utf-8").readlines()
v5 = [json.loads(l) for l in v5_lines]

# Cap chain_score at 1.0 (fix known LLM over-scoring bug)
for r in v3:
    r["chain_score"] = min(r["chain_score"], 1.0)
for r in v5:
    r["chain_score"] = min(r["chain_score"], 1.0)

v3_map = {r["id"]: r for r in v3}
v5_map = {r["id"]: r for r in v5}

# Per-question comparison
print("=" * 110)
hdr = f"{'Q#':>3} {'v3_CoT':>8} {'v5_HyDE':>8} {'Diff':>7} {'v3_xb':>5} {'v5_xb':>5} {'v5_lat':>7} | Question"
print(hdr)
print("-" * 110)

improved = degraded = unchanged = 0
for i, qid in enumerate(v3_map.keys(), 1):
    r3 = v3_map[qid]
    r5 = v5_map.get(qid)
    if r5 is None:
        continue
    diff = r5["chain_score"] - r3["chain_score"]
    if diff > 0.01:
        improved += 1
        marker = " ▲"
    elif diff < -0.01:
        degraded += 1
        marker = " ▼"
    else:
        unchanged += 1
        marker = "  "
    print(
        f"{i:3d}  {r3['chain_score']:8.3f} {r5['chain_score']:8.3f} {diff:+7.3f}{marker}"
        f" {int(r3['cross_book_hit']):>5} {int(r5['cross_book_hit']):>5}"
        f" {r5['latency_s']:7.1f} | {r3['question'][:40]}"
    )

print("-" * 110)

# Summary stats
v3_scores = [r["chain_score"] for r in v3]
v5_scores = [r["chain_score"] for r in v5]
n = len(v3_scores)

v3_mean = sum(v3_scores) / n
v5_mean = sum(v5_scores) / n
v3_hopok = sum(1 for s in v3_scores if s >= 0.6) / n
v5_hopok = sum(1 for s in v5_scores if s >= 0.6) / n
v3_fullok = sum(1 for s in v3_scores if s >= 1.0) / n
v5_fullok = sum(1 for s in v5_scores if s >= 1.0) / n
v3_cross = sum(1 for r in v3 if r["cross_book_hit"]) / n
v5_cross = sum(1 for r in v5 if r["cross_book_hit"]) / n
v3_p50 = sorted(r["latency_s"] for r in v3)[n // 2]
v5_p50 = sorted(r["latency_s"] for r in v5)[n // 2]

print(f"\nPer-question: improved={improved}  degraded={degraded}  unchanged={unchanged}")
print()
print(f"{'Metric':>20} {'v3_CoT':>10} {'v5_HyDE+G':>10} {'Delta':>10}")
print("-" * 55)
print(f"{'chain_score':>20} {v3_mean:>10.4f} {v5_mean:>10.4f} {v5_mean - v3_mean:>+10.4f}")
print(f"{'hop_ok(>=0.6)':>20} {v3_hopok:>9.1%} {v5_hopok:>9.1%} {v5_hopok - v3_hopok:>+9.1%}")
print(f"{'full_ok(=1.0)':>20} {v3_fullok:>9.1%} {v5_fullok:>9.1%} {v5_fullok - v3_fullok:>+9.1%}")
print(f"{'cross_book_hit':>20} {v3_cross:>9.1%} {v5_cross:>9.1%} {v5_cross - v3_cross:>+9.1%}")
print(f"{'p50_latency':>20} {v3_p50:>9.1f}s {v5_p50:>9.1f}s {v5_p50 - v3_p50:>+9.1f}s")

# Show biggest improvements and degradations
print("\n--- Top 5 Improvements (HyDE helped) ---")
deltas = []
for qid in v3_map:
    r3 = v3_map[qid]
    r5 = v5_map.get(qid)
    if r5:
        deltas.append((r5["chain_score"] - r3["chain_score"], r3["question"][:60], r3["chain_score"], r5["chain_score"]))
deltas.sort(reverse=True)
for d, q, s3, s5 in deltas[:5]:
    print(f"  {d:+.3f}  ({s3:.3f} -> {s5:.3f})  {q}")

print("\n--- Top 5 Degradations (HyDE hurt) ---")
for d, q, s3, s5 in deltas[-5:]:
    print(f"  {d:+.3f}  ({s3:.3f} -> {s5:.3f})  {q}")
