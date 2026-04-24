"""Analyze why HyDE works well alone but doesn't help Graph RAG."""
import json, os

# Load all three result sets
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

with open(f'benchmarks/results/v8/{v3_files[-1]}', encoding='utf-8') as f:
    v3 = json.load(f)
with open(f'benchmarks/results/v8/{v5_files[-1]}', encoding='utf-8') as f:
    v5 = json.load(f)
with open(hyde_files[-1], encoding='utf-8') as f:
    hyde = json.load(f)

n = len(v3['records'])

# --- 1. Pipeline architecture comparison ---
print("=" * 70)
print("1. 管道架构对比")
print("=" * 70)
print("""
纯 HyDE:     问题 → HyDE伪文档 → 向量搜索(k=15) → BGE rerank(top_n=5) → 5篇
Graph v3:    问题 → 向量搜索(k=10 seed) → 图BFS扩展(~30邻居) → BGE rerank(top_n=7) → 7篇
HyDE+Graph:  问题 → HyDE伪文档 → 向量搜索(k=10 seed) → 图BFS扩展(~30邻居) → BGE rerank(top_n=7) → 7篇
""")

# --- 2. Context overlap analysis ---
print("=" * 70)
print("2. 上下文重叠分析")
print("=" * 70)
print(f"{'Q':>2}  {'h_n':>3} {'v3_n':>4} {'v5_n':>4}  {'v3∩h':>4} {'v5∩h':>4} {'v5∩v3':>5} {'v5独有':>6}  问题")

total_v5_v3 = 0
total_v5_hyde = 0
total_v5_new = 0
total_v3_hyde = 0
for i in range(n):
    h_set = set(c[:80] for c in hyde['records'][i]['retrieved_contexts'])
    v3_set = set(c[:80] for c in v3['records'][i]['retrieved_contexts'])
    v5_set = set(c[:80] for c in v5['records'][i]['retrieved_contexts'])
    v3h = len(v3_set & h_set)
    v5h = len(v5_set & h_set)
    v5v3 = len(v5_set & v3_set)
    v5new = len(v5_set - v3_set - h_set)
    total_v5_v3 += v5v3
    total_v5_hyde += v5h
    total_v5_new += v5new
    total_v3_hyde += v3h
    q = v3['records'][i]['question'][:35]
    print(f"{i+1:>2}   {len(h_set):>2}   {len(v3_set):>3}   {len(v5_set):>3}    {v3h:>3}   {v5h:>3}   {v5v3:>3}     {v5new:>3}   {q}")

print(f"\n汇总:")
print(f"  v3 与 hyde 重叠: {total_v3_hyde}/{n*7} ({total_v3_hyde/(n*7):.1%})")
print(f"  v5 与 hyde 重叠: {total_v5_hyde}/{n*7} ({total_v5_hyde/(n*7):.1%})")
print(f"  v5 与 v3  重叠: {total_v5_v3}/{n*7} ({total_v5_v3/(n*7):.1%})")
print(f"  v5 独有(非v3亦非hyde): {total_v5_new}/{n*7} ({total_v5_new/(n*7):.1%})")

# --- 3. Key insight: reranker is the real differentiator ---
print("\n" + "=" * 70)
print("3. 候选池大小 vs 最终精度")
print("=" * 70)

# Plain HyDE: 15 candidates → 5 kept = 33% keep rate (strict filtering)
# Graph: 10 seeds + ~30 neighbors = ~40 candidates → 7 kept = 17.5% keep rate
# But graph candidates are topology-based, not embedding-based
print(f"""
纯 HyDE:     15个候选 → 保留5个 (33%保留率, 从向量相似的候选中精选)
Graph v3/v5: ~40个候选(10 seed + ~30 graph邻居) → 保留7个 (17.5%保留率)

关键差异：
  - HyDE 的15个候选全部来自向量相似度搜索（语义相关性高）
  - Graph 的~40个候选中只有10个是向量搜索的，其余30个是图扩展的拓扑邻居
  - 图邻居基于"共享术语"边选出，不保证与问题语义相关
  - reranker 在两种管道中都用原始问题打分，但候选池质量不同
""")

# --- 4. Per-question: where does HyDE change graph seeds? ---
print("=" * 70)
print("4. v5 vs v3 逐题对比（HyDE改变了多少seed？）")
print("=" * 70)
print(f"{'Q':>2}  {'v3延迟':>6} {'v5延迟':>6} {'差值':>5}  {'重叠':>4} {'v5新增':>5}  {'解读':>6}")

better = 0
worse = 0
same = 0
for i in range(n):
    v3r = v3['records'][i]
    v5r = v5['records'][i]
    v3_set = set(c[:80] for c in v3r['retrieved_contexts'])
    v5_set = set(c[:80] for c in v5r['retrieved_contexts'])
    overlap = len(v5_set & v3_set)
    new = len(v5_set - v3_set)
    
    lat_v3 = v3r['latency_total']
    lat_v5 = v5r['latency_total']
    diff = lat_v5 - lat_v3
    
    if new == 0:
        label = "无变化"
        same += 1
    elif new <= 2:
        label = "微调"
        same += 1
    else:
        label = "显著变化"
    
    print(f"{i+1:>2}  {lat_v3:>6.1f} {lat_v5:>6.1f} {diff:>+5.1f}   {overlap}/7   {new}/7   {label}")

# --- 5. Answer length comparison ---
print("\n" + "=" * 70)
print("5. 回答长度对比")
print("=" * 70)
v3_lens = [len(r['answer']) for r in v3['records']]
v5_lens = [len(r['answer']) for r in v5['records']]
h_lens  = [len(r['answer']) for r in hyde['records']]
print(f"  纯 HyDE 平均回答长度: {sum(h_lens)/len(h_lens):.0f} 字")
print(f"  Graph v3 平均回答长度: {sum(v3_lens)/len(v3_lens):.0f} 字")
print(f"  HyDE+Graph v5 平均回答长度: {sum(v5_lens)/len(v5_lens):.0f} 字")

# --- 6. Conclusion ---
print("\n" + "=" * 70)
print("6. 核心结论")
print("=" * 70)
print("""
纯 HyDE 表现好的原因：
  1. HyDE 解决"词汇鸿沟"：现代汉语问题 → 古典汉语伪文档 → 与古籍embedding更匹配
  2. BGE reranker 是"降噪器"：reranker用原始问题（不是HyDE文本）打分，
     过滤掉HyDE偏题幻觉引入的无关文档
  3. 候选池纯净：15个候选全部来自向量搜索，语义质量均匀

HyDE+Graph v5 表现差的真正原因（不是"HyDE引入噪声"）：
  1. 架构不兼容：HyDE优化的是embedding检索，但Graph扩展用的是拓扑遍历，
     直接覆盖了HyDE的改善
  2. HyDE只影响10个seed的选择，但Graph又往里塞了~30个拓扑邻居，
     seed的微小改善被淹没
  3. 数据证明：v5与v3重叠65%，v5与纯HyDE重叠仅0.6% → 
     HyDE改变了一些seed，但图扩展重新主导了最终结果
  4. reranker从~40个混合候选(向量+拓扑)中选7个，不如从15个纯向量候选中选5个精准
""")
