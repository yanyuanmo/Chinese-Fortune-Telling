import json

with open('benchmarks/results/multihop/20260417_211506/results_graph_rag_v6_mn15.json', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total: {len(data)} samples")
scores = [d['chain_score'] for d in data]
print(f"Score distribution:")
from collections import Counter
for s, cnt in sorted(Counter([round(x, 3) for x in scores]).items()):
    print(f"  {s:.3f}: {cnt}x")

print()
print("First 10:")
for i, d in enumerate(data[:10]):
    ans = d.get('answer', '')[:80]
    cs = d['chain_score']
    lat = d.get('latency', 0)
    print(f"Q{i+1}: chain={cs:.3f}  lat={lat:.1f}s  ans={ans}")

print()
# Check for error patterns
errors = [i for i, d in enumerate(data) if d['chain_score'] == 0]
print(f"Zero-score questions: {len(errors)} / {len(data)}")
if errors:
    for i in errors[:5]:
        print(f"  Q{i+1}: {data[i].get('answer', '')[:100]}")
