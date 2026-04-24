import json, glob

results = {}
# v3 CoT (baseline)
cot_files = sorted(glob.glob('benchmarks/results/multihop/*/results_graph_rag_v3_cot_only.json'))
with open(cot_files[-1], encoding='utf-8') as f:
    results['v3_CoT (mn=30)'] = json.load(f)

# vf50
with open('benchmarks/results/multihop/20260417_214751/results_graph_rag_v7_vf50.json', encoding='utf-8') as f:
    results['v7_vf50'] = json.load(f)

# vf100
with open('benchmarks/results/multihop/20260417_214751/results_graph_rag_v7_vf100.json', encoding='utf-8') as f:
    results['v7_vf100'] = json.load(f)

# mn10
with open('benchmarks/results/multihop/20260417_211506/results_graph_rag_v6_mn10.json', encoding='utf-8') as f:
    results['v6_mn10'] = json.load(f)

print('=== Summary (chain_score capped at 1.0) ===')
header = f"{'Config':20s}  chain   hop_ok  full_ok  cross   p50"
print(header)
for name, data in results.items():
    capped = [min(d['chain_score'], 1.0) for d in data]
    mean = sum(capped)/len(capped)
    hop_ok = sum(1 for s in capped if s >= 0.6)/len(capped)
    full_ok = sum(1 for s in capped if s >= 0.99)/len(capped)
    cross = sum(d.get('cross_book_hit',1) for d in data)/len(data)
    lats = [d.get('latency',0) for d in data]
    p50 = sorted(lats)[len(lats)//2] if lats else 0
    print(f"{name:20s}  {mean:.4f}  {hop_ok:.4f}  {full_ok:.4f}   {cross:.4f}  {p50:.1f}s")

print()
print('=== Per-question comparison (v3CoT vs vf50 vs vf100) ===')
print('Q   v3CoT   vf50    vf100   d(50)   d(100)')
b50 = w50 = s50 = b100 = w100 = s100 = 0
for i in range(36):
    s1 = min(results['v3_CoT (mn=30)'][i]['chain_score'], 1.0)
    s2 = min(results['v7_vf50'][i]['chain_score'], 1.0)
    s3 = min(results['v7_vf100'][i]['chain_score'], 1.0)
    d2 = s2 - s1
    d3 = s3 - s1
    f2 = ' +' if d2>0.01 else (' -' if d2<-0.01 else '  ')
    f3 = ' +' if d3>0.01 else (' -' if d3<-0.01 else '  ')
    if d2>0.01: b50 += 1
    elif d2<-0.01: w50 += 1
    else: s50 += 1
    if d3>0.01: b100 += 1
    elif d3<-0.01: w100 += 1
    else: s100 += 1
    print(f"{i+1:>2}  {s1:.3f}   {s2:.3f}   {s3:.3f}  {d2:>+.3f}{f2} {d3:>+.3f}{f3}")

print(f"\nvf50  Better: {b50}  Worse: {w50}  Same: {s50}")
print(f"vf100 Better: {b100}  Worse: {w100}  Same: {s100}")
