"""
Compare all v8 multihop benchmark results side by side.
"""
import json, glob, os

# Find all multihop result dirs
base = "benchmarks/results/multihop"

# Map config names to result files
configs_of_interest = {
    "graph_v3_CoT_small":    None,  # bge-small + reranker-base (baseline)
    "hyde_rerank_topn7":     None,  # HyDE bge-small
    "graph_v8_bge_base":     None,  # bge-base + reranker-base
    "hyde_v8_bge_base":      None,  # HyDE bge-base
    "graph_v8_reranker_lg":  None,  # bge-small + reranker-large
}

# Scan all result dirs
for d in sorted(glob.glob(os.path.join(base, "*/"))):
    for f in glob.glob(os.path.join(d, "results_*.json")):
        fname = os.path.basename(f)
        if "graph_rag_v3_cot_only" in fname:
            configs_of_interest["graph_v3_CoT_small"] = f
        elif "hyde_rerank_topn7" in fname and "v8" not in fname:
            configs_of_interest["hyde_rerank_topn7"] = f
        elif "graph_rag_v8_bge_base" in fname:
            configs_of_interest["graph_v8_bge_base"] = f
        elif "hyde_rerank_v8_bge_base" in fname:
            configs_of_interest["hyde_v8_bge_base"] = f
        elif "graph_rag_v8_reranker_large" in fname:
            configs_of_interest["graph_v8_reranker_lg"] = f

print("=== Found result files ===")
for name, path in configs_of_interest.items():
    print(f"  {name}: {path or 'NOT FOUND'}")
print()

# Load and compare
results = {}
for name, path in configs_of_interest.items():
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as fp:
            data = json.load(fp)
        scores = [min(r["chain_score"], 1.0) for r in data]
        cross = [r["cross_book_hit"] for r in data]
        n = len(data)
        results[name] = {
            "data": data,
            "scores": scores,
            "chain": sum(scores) / n,
            "hop_ok": sum(1 for s in scores if s >= 0.6) / n,
            "full_ok": sum(1 for s in scores if s >= 0.99) / n,
            "cross_hit": sum(cross) / n,
            "n": n,
        }

print("=== Summary ===")
print(f"{'Config':<25} {'chain':>7} {'hop_ok':>7} {'full_ok':>8} {'cross':>7} {'n':>3}")
print("-" * 65)
for name in configs_of_interest:
    if name in results:
        r = results[name]
        print(f"{name:<25} {r['chain']:>7.4f} {r['hop_ok']:>7.4f} {r['full_ok']:>8.4f} {r['cross_hit']:>7.4f} {r['n']:>3}")

# Per-question comparison
baseline_name = "graph_v3_CoT_small"
if baseline_name in results:
    baseline = results[baseline_name]
    print(f"\n=== Per-question vs {baseline_name} (chain={baseline['chain']:.4f}) ===")
    
    for other_name in configs_of_interest:
        if other_name == baseline_name or other_name not in results:
            continue
        other = results[other_name]
        better = worse = same = 0
        delta_sum = 0
        for i in range(min(baseline["n"], other["n"])):
            b = baseline["scores"][i]
            o = other["scores"][i]
            d = o - b
            delta_sum += d
            if d > 0.01:
                better += 1
            elif d < -0.01:
                worse += 1
            else:
                same += 1
        n = min(baseline["n"], other["n"])
        print(f"  {other_name}: Better={better}  Worse={worse}  Same={same}  "
              f"Δchain={delta_sum/n:+.4f}")
