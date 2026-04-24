"""Gather all normal-dataset benchmark results for comparison."""
import json
from pathlib import Path

results_dir = Path("benchmarks/results")

# Collect all detail files
files = [
    ("v5/hyde_rerank_bge", "v5/hyde_rerank_bge_20260222_174736_detail.json"),
    ("v5b/hyde_rerank_bge", "v5b/hyde_rerank_bge_20260222_182009_detail.json"),
    ("v5b/hyde_rerank_k20", "v5b/hyde_rerank_k20_20260222_183057_detail.json"),
    ("v5b/hyde_rerank_multi2", "v5b/hyde_rerank_multi2_20260222_184239_detail.json"),
    ("v5d/hyde_rerank_bge", "v5d/hyde_rerank_bge_20260222_185348_detail.json"),
    ("v5d/hyde_rerank_topn7", "v5d/hyde_rerank_topn7_20260222_190538_detail.json"),
    ("v5e/hyde_rerank_topn7", "v5e/hyde_rerank_topn7_20260222_191831_detail.json"),
    ("v5e/hyde_rerank_topn9", "v5e/hyde_rerank_topn9_20260222_193129_detail.json"),
]

print(f"{'Config':<30} {'faith':>8} {'relev':>8} {'recall':>8} {'prec':>8} {'p50':>8} {'n':>4}")
print("-" * 80)

for label, fpath in files:
    full = results_dir / fpath
    if not full.exists():
        print(f"{label:<30} NOT FOUND")
        continue
    data = json.load(open(full, encoding="utf-8"))
    s = data["summary"]
    print(
        f"{label:<30}"
        f" {s.get('faithfulness', 0):>8.4f}"
        f" {s.get('answer_relevancy', 0):>8.4f}"
        f" {s.get('context_recall', 0):>8.4f}"
        f" {s.get('context_precision', 0):>8.4f}"
        f" {s.get('latency_p50', 0):>8.1f}"
        f" {s.get('num_samples', 0):>4}"
    )

# Also check for hybrid_rerank results
print("\n--- hybrid_rerank baselines ---")
hrr_files = sorted(results_dir.glob("hybrid_rerank_bge*_detail.json"))
for f in hrr_files:
    data = json.load(open(f, encoding="utf-8"))
    s = data["summary"]
    label = s.get("config", f.stem[:30])
    print(
        f"{label:<30}"
        f" {s.get('faithfulness', 0):>8.4f}"
        f" {s.get('answer_relevancy', 0):>8.4f}"
        f" {s.get('context_recall', 0):>8.4f}"
        f" {s.get('context_precision', 0):>8.4f}"
        f" {s.get('latency_p50', 0):>8.1f}"
        f" {s.get('num_samples', 0):>4}"
    )
