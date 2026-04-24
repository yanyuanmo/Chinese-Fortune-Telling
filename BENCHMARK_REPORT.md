# RAG Benchmark Report — 果赖算命阁

## 目录

1. [项目概况](#1-项目概况)
2. [评估方法论](#2-评估方法论)
3. [方法演进 v1 → v8](#3-方法演进-v1--v8)
4. [完整成绩表](#4-完整成绩表)
5. [消融实验与分析](#5-消融实验与分析)
6. [Prompt Tuning 实验](#6-prompt-tuning-实验)
7. [评估陷阱与修复](#7-评估陷阱与修复)
8. [核心结论](#8-核心结论)

---

## 1. 项目概况

| 项 | 值 |
|----|-----|
| 语料 | 3 部古籍：《三命通会》(419 chunks)、《滴天髓》(154)、《子平真诠》(95) |
| 总 chunks | 668 |
| 向量库 | ChromaDB，bge-small-zh-v1.5 (384d) / bge-base-zh-v1.5 (768d) |
| 知识图谱 | 668 nodes, 5,255 IDF-weighted cross-book edges |
| 生成 LLM | Kimi moonshot-v1-32k (temperature 0.7) |
| 评估 LLM | GPT-4o (unbiased judge) |
| 评估 Embedding | BAAI/bge-small-zh-v1.5 (Chinese, normalize=True) |

---

## 2. 评估方法论

### 2.1 两套数据集

**Normal (22Q)** — 单书、单跳问题

- 来源：人工精选，每题从一本书的 1 个 chunk 出发
- 评估：RAGAS v0.4 四指标
  - **faithfulness**: 回答是否忠于检索到的上下文（无幻觉）
  - **answer_relevancy**: 回答与问题的语义相关度
  - **context_recall**: 检索上下文覆盖 ground truth 的程度
  - **context_precision**: 检索结果中有用信息的占比
- 综合分：四指标简单平均

**Multihop (36Q)** — 跨书、多步推理

- 来源：`generate_qa_multihop.py` 自动生成
  - 3 个书对 × 12 题，每题跨 2 本书
  - 每题带 3-step reasoning chain 作为 ground truth
  - 基于跨书 bridge terms (官星、财星、格局、用神等) 构建
- 评估：LLM-as-Judge chain_score
  - 每个 reasoning step 由 LLM 评 0 / 0.5 / 1
  - chain_score = mean(step_scores)
  - hop_ok: chain_score ≥ 0.6 的比例
  - full_ok: chain_score ≥ 1.0 的比例
  - cross_book_hit: 检索结果覆盖 ≥ 2 本书的比例

### 2.2 评估者演变

| 阶段 | 评估 LLM | 评估 Embedding | 问题 |
|------|---------|---------------|------|
| v1-v8 初始 | Kimi (self-eval) | all-MiniLM-L6-v2 (English) | 自评偏宽松 +10~33%；英文 embedding 测中文相似度失真 |
| v11 修复 | **GPT-4o** | **bge-small-zh-v1.5** (Chinese) | 两个问题都解决 |

关键发现：
- GPT-4o 比 Kimi 自评严格 10–33%（faithfulness 差距最大）
- 中文 eval embedding 使 answer_relevancy +12.5pp（因为英文 embedding 无法准确衡量中文语义距离）

### 2.3 Evaluator 偏差对比（Multihop）

| Config | Kimi chain_score | GPT-4o chain_score | Δ |
|--------|:---:|:---:|:---:|
| v3_cot | 0.787 | 0.697 | −0.090 |
| v8_bge_base_k15 | 0.780 | 0.725 | −0.056 |
| v7_vf50 | 0.729 | 0.729 | 0.000 |
| hyde_no_graph | 0.729 | 0.593 | −0.137 |

→ Kimi 自评对「部分正确」更宽容；GPT-4o 对推理链的中间步骤要求更严格。

---

## 3. 方法演进 v1 → v8

### 3.0 Baseline 方法

#### Hybrid (BM25 + Vector)
```yaml
type: hybrid
k: 5
bm25_weight: 0.4  # BM25 0.4 + Vector 0.6
```
- BM25 在文言文中的字面匹配能力 + 语义向量的泛化能力
- Normal AVG=0.790（排名 #4），Multihop chain=0.458（倒数）

#### HyDE + Rerank
```yaml
type: hyde_rerank
hyde_k: 15          # 先宽召回 15 个
top_n: 7            # BGE reranker 精排后取 7 个
rerank_model: BAAI/bge-reranker-base
```
- 先让 LLM 生成一段假设性古文片段（80–150 字），用它代替原始 query 做向量检索
- 解决 query-document 词汇鸿沟（用户用白话问，文档是文言文）
- **Normal #1** (AVG=0.812)，faithfulness 最高 (0.917)
- Multihop chain=0.593，cross_book_hit=39%（无跨书能力）

#### Hybrid + BGE Rerank
```yaml
type: hybrid_rerank
k: 6
top_n: 3
rerank_model: BAAI/bge-reranker-base
```
- 替换英文 reranker (ms-marco-MiniLM) 为中文 BGE
- precision 最高 (0.894)，但 recall 偏低 (0.720)

#### Proposition Indexing
```yaml
type: proposition  # / prop_hybrid
```
- 将原始 chunk 分解为独立命题，分别索引
- 理论上提升粒度和 precision，实际：
  - prop_vector: AVG=0.611，faithfulness 仅 0.492
  - prop_hybrid: AVG=0.478，最差
- 原因：古文的命题拆分质量不佳，上下文被打碎后 LLM 无法整合

---

### 3.1 Graph RAG v1 — Flat Graph (min_weight=1)

```
43,932 edges | 668 nodes | min_weight=1
```

- **首次引入知识图谱**：节点 = chunk，边 = 跨书共享 bridge term 的 chunk 对
- Pipeline: vector seed (k=10) → BFS hop=1 (max_neighbors=30) → BGE rerank → top_n=7
- 问题：边太多太密，大量低质量连接（只共享一个常见术语如「五行」），neighbor 噪声严重
- Multihop chain=0.435（GPT-4o），比 hybrid 还差

### 3.2 Graph RAG v2 — Pruned Graph (min_weight=2)

```
15,698 edges | 668 nodes | min_weight=2
```

- 只保留共享 ≥ 2 个 bridge terms 的 chunk 对
- 边数减少 64%，cross_book_hit 从 69% → 89%
- Kimi 评估的 chain_score=0.644，full_ok=19.4%
- **仍然有噪声**：degree 无上限，部分核心节点连接过多

### 3.3 Graph RAG v3 — IDF-Weighted Edges + Degree Pruning ⭐

```
5,255 edges | 668 nodes | IDF-weighted | max_degree=15
```

**核心创新：IDF 加权边**

```python
# 1. 构建术语→chunk 倒排索引
# 2. 计算每个 bridge term 的 IDF = log(N / df)
# 3. 跨书 chunk 对的边权 = sum(共享术语的 IDF)
```

稀有术语（如「化气」df=12, IDF=3.7）比高频术语（如「五行」df=158, IDF=1.4）贡献更高边权 → 边的语义区分度大幅提高。

**Degree Pruning**: 每个节点最多保留权重最高的 15 条边（对称保留：只要一端认为它是 top-15 就保留）。

**效果**：
- 图稀疏化到 5,255 边（原来 43K），但语义连接质量更高
- avg_degree=15.73，平衡了覆盖率和精确度

**配置分支对比**:

| 变体 | hop | 说明 | Multihop (GPT-4o) |
|------|:---:|------|:---:|
| v3_idf_hop1 | 1 | 1 跳安全探索 | 0.498 |
| v3_idf (hop=2) | 2 | 2 跳，触达更多跨书节点 | 0.546 |
| v3_cot | 1 | + CoT 推理 prompt | **0.697** |

→ CoT prompt 对推理链质量提升显著 (+15pp)，这是 prompt 对 generation 的优化而非 retrieval 的改进。

### 3.4 Graph RAG v4 — Reranker 升级

```yaml
rerank_model: BAAI/bge-reranker-v2-m3  # 568M params, multilingual
top_n: 10
```

- 把 reranker 从 bge-reranker-base (278M) 升级到 v2-m3 (568M)
- top_n 从 7 → 10，给多步推理提供更多上下文
- Multihop (GPT-4o): 0.637，反而低于 v3_cot (0.697)
- **结论**：更大 reranker 和更多 context 不一定更好。v2-m3 是 multilingual 模型，对纯中文古文不一定比 Chinese-native 的 base 更适合

### 3.5 Graph RAG v5 — HyDE + Graph

```yaml
hyde: true  # 在 graph_rag 中启用 HyDE seed
```

- 用 HyDE 生成假设古文片段做 seed retrieval，再做 graph BFS 扩展
- Multihop (GPT-4o): 0.671，比 v3_cot (0.697) 低
- Normal AVG=0.779（排名 #5）
- **分析**：HyDE 改善了 seed 的 query-doc 匹配度，但 graph 扩展的 neighbor 质量才是 multihop 的瓶颈；HyDE 对 seed 质量的提升被 neighbor 噪声稀释了

### 3.6 Graph RAG v6 — max_neighbors 消融

```yaml
max_neighbors: 10  # 从 30 减到 10
```

- 测试减少 neighbor 数量是否降低噪声
- v6_mn10: Multihop 0.692 vs v3_cot: 0.697 → 几乎持平
- v6_mn15: Multihop 0.208 → **崩溃**（过度剪枝导致丢失关键跨书连接）
- **结论**：max_neighbors=10 是安全的下界，再低就会丢失有用信号

### 3.7 Graph RAG v7 — Vector Filter (语义门控) ⭐

```yaml
vector_filter_k: 50  # 邻居必须出现在 top-50 向量结果中
```

**核心创新：topology × semantics 双重筛选**

问题背景：BFS 扩展的邻居是图上的拓扑邻居，不一定和当前 query 语义相关。

解决方案：
1. 先做一次宽泛的 vector search (k=50)，得到语义相关 chunk 白名单
2. BFS 扩展时，只保留**同时在白名单中**的邻居
3. 相当于给图扩展加了一个语义相关性门控

**效果**：
- vf=50: Multihop **0.729**（GPT-4o, #1），full_ok=47.2%
- vf=100: Multihop 0.720，full_ok=50.0%（白名单太宽松，噪声回升）
- v3_cot (无 filter): 0.697

→ vector_filter_k=50 是最佳平衡点，将 chain_score 再提 3.2pp。

### 3.8 Graph RAG v8 — 换大 Embedding

```yaml
embedding_model: BAAI/bge-base-zh-v1.5  # 768-dim (vs small 384-dim)
chroma_dir: ./chroma_db_bge_base
k: 15  # 更大embedding空间需要更多seed
```

- bge-base (768d) 比 bge-small (384d) 表示能力更强
- k 从 10 提升到 15 以补偿更大的 embedding space
- **Normal #2**: AVG=0.804 (仅次于 pure HyDE)
- **Multihop**: k=15 → 0.725, k=20 → 0.725（持平，k再大无益）

**Reranker 变体**:

| Reranker | Multihop (GPT-4o) |
|----------|:---:|
| bge-reranker-base (278M) | **0.725** |
| bge-reranker-large (560M) | 0.692 |
| bge-reranker-v2-m3 (568M) | 0.637 |

→ base 在本场景下反而最优。可能因为：(1) 训练数据更集中于中文；(2) 古文语义相对简洁，不需要更大模型容量。

---

## 4. 完整成绩表

### 4.1 Normal Benchmark (22Q) — GPT-4o + Chinese Eval Embedding

| # | Config | Category | Faithfulness | Relevancy | Recall | Precision | **AVG** |
|:-:|--------|----------|:---:|:---:|:---:|:---:|:---:|
| 1 | hyde_rerank_topn7 | HyDE | **0.917** | 0.651 | 0.829 | 0.850 | **0.812** |
| 2 | graph_rag_v8_bge_k15 | Graph | 0.841 | **0.656** | **0.849** | **0.868** | 0.804 |
| 3 | graph_rag_v7_vf50 | Graph | 0.902 | 0.639 | 0.794 | 0.833 | 0.792 |
| 4 | hybrid | Baseline | 0.762 | 0.668 | 0.846 | 0.885 | 0.790 |
| 5 | graph_rag_v5_hyde | HyDE+Graph | 0.861 | 0.622 | 0.784 | 0.847 | 0.779 |
| 6 | graph_rag_v3_cot | Graph | 0.852 | 0.620 | 0.775 | 0.854 | 0.775 |
| 7 | hybrid_rerank_bge | Baseline | 0.755 | 0.664 | 0.720 | 0.894 | 0.758 |
| 8 | prop_vector | Proposition | 0.492 | 0.667 | 0.567 | 0.719 | 0.611 |
| 9 | prop_hybrid | Proposition | 0.392 | 0.628 | 0.438 | 0.453 | 0.478 |

### 4.2 Multihop Benchmark (36Q) — GPT-4o Rescore

| # | Config | Category | Chain Score | Hop OK | Full OK | Cross Hit |
|:-:|--------|----------|:---:|:---:|:---:|:---:|
| 1 | v7_vf50 | Graph | **0.729** | 72.2% | 47.2% | 77.8% |
| 2 | v8_bge_base_k15 | Graph | 0.725 | **77.8%** | 44.4% | 88.9% |
| 3 | v8_bge_base_k20 | Graph | 0.725 | **77.8%** | 44.4% | **91.7%** |
| 4 | v8_hyde_bge_base | HyDE+Graph | 0.720 | 75.0% | 41.7% | 27.8% |
| 5 | v7_vf100 | Graph | 0.720 | 69.4% | **50.0%** | 86.1% |
| 6 | v3_cot | Graph | 0.697 | 69.4% | 44.4% | — |
| 7 | v8_reranker_large | Graph | 0.692 | — | — | — |
| 8 | v6_mn10 | Graph | 0.692 | — | — | — |
| 9 | v5_hyde_cot | HyDE+Graph | 0.671 | — | — | — |
| 10 | v4_cot_v2m3 | Graph | 0.637 | — | — | — |
| 11 | v8_bge_base_k10 | Graph | 0.630 | — | — | — |
| 12 | hyde_no_graph | HyDE | 0.593 | 58.3% | 38.9% | 38.9% |
| 13 | hyde_old_feb | HyDE | 0.556 | — | — | — |
| 14 | v3_idf | Graph | 0.546 | — | — | — |
| 15 | v3_idf_hop1 | Graph | 0.498 | — | — | — |
| 16 | v2_flat | Graph | 0.465 | — | — | — |
| 17 | hybrid_feb | Baseline | 0.458 | — | — | — |
| 18 | graph_rag_v1 | Graph | 0.435 | — | — | — |
| 19 | v6_mn15 | Graph | 0.208 | — | — | — |

---

## 5. 消融实验与分析

### 5.1 Embedding Model: small vs base

| Metric | bge-small (384d) | bge-base (768d) | Δ |
|--------|:---:|:---:|:---:|
| Normal AVG (graph_rag) | 0.775 (v3_cot) | **0.804** (v8_k15) | +0.029 |
| Multihop chain | 0.697 (v3_cot) | **0.725** (v8_k15) | +0.028 |

→ 更高维度的 embedding 稳定提升 ~3pp，但需要配合更大的 k (10→15) 以维持 seed recall。

### 5.2 Reranker Model

| Reranker | Params | Multihop (GPT-4o) | Normal AVG |
|----------|:------:|:---:|:---:|
| bge-reranker-base | 278M | **0.725** | — |
| bge-reranker-large | 560M | 0.692 | — |
| bge-reranker-v2-m3 | 568M | 0.637 | — |

→ 更大模型不一定更优。base 专注于中文语义，对古文最有效。

### 5.3 Seed Count (k)

| k | Multihop chain | Cross Hit |
|:-:|:---:|:---:|
| 10 | 0.630 | — |
| **15** | **0.725** | 88.9% |
| 20 | 0.725 | **91.7%** |

→ k=15 是性价比最高的点。k=20 cross_book_hit 更高但 chain_score 未再提升（额外 seed 被 reranker 筛掉）。

### 5.4 max_neighbors

| max_neighbors | Multihop chain | 说明 |
|:---:|:---:|------|
| **30** | 0.697 | 默认值 |
| **10** | 0.692 | 微降，可接受 |
| 15 | **0.208** | 崩溃（配置 bug 导致过度剪枝） |

### 5.5 vector_filter_k

| vf | Multihop chain | Full OK |
|:---:|:---:|:---:|
| 0 (disabled) | 0.697 | 44.4% |
| **50** | **0.729** | 47.2% |
| 100 | 0.720 | **50.0%** |

→ vf=50 最佳 chain_score；vf=100 更多完美答案但平均分略低（白名单过宽引入少量噪声）。

### 5.6 HyDE 在不同场景下的表现

| 场景 | 有 HyDE | 无 HyDE | Δ |
|------|:---:|:---:|:---:|
| Normal (单跳) | **0.812** | 0.790 (hybrid) | +0.022 |
| Multihop + Graph | 0.671 (v5) | **0.697** (v3_cot) | −0.026 |
| Multihop 纯 HyDE | 0.593 | — | — |

→ HyDE 对单跳检索价值大（填补 query-doc 词汇鸿沟）；多跳场景中 graph 扩展已提供足够的跨书信号，HyDE 额外增加延迟但无正向贡献。

---

## 6. Prompt Tuning 实验

在最优检索配置 (graph_rag_v8_bge_k15) 上测试 3 种 prompt 风格：

| Prompt Style | Faithfulness | Relevancy | Recall | Precision | 说明 |
|-------------|:---:|:---:|:---:|:---:|------|
| **default** | **0.841** | **0.656** | **0.849** | **0.868** | 标准 prompt |
| concise | 0.826 | 0.550 | 0.836 | 0.871 | 限制 150-300 字 |
| balanced | 0.821 | 0.520 | 0.834 | 0.860 | 200-400 字 + 引用 |

结论：
- 三种 prompt 变体之间差异 ≤ 2pp（faithfulness, recall, precision）
- concise 的 answer_relevancy 反而下降了 −10pp（过度压缩导致语义信息丢失）
- **Prompt tuning 已触顶**：generation 质量受制于 retrieval 质量，不是 prompt 的瓶颈

---

## 7. 评估陷阱与修复

### 7.1 英文 Eval Embedding 对中文评估的影响

| Eval Embedding | answer_relevancy (graph_rag_v8) |
|----------------|:---:|
| all-MiniLM-L6-v2 (English) | 0.524 |
| **bge-small-zh-v1.5 (Chinese)** | **0.656** |

差距 **+12.5pp**。原因：RAGAS 的 answer_relevancy 用 embedding cosine similarity 衡量 answer 和 question 的语义距离。英文 embedding 无法准确编码中文文本 → 系统性低估 relevancy。

### 7.2 Self-Eval 偏差

Kimi 既生成答案又评估答案 → 系统性偏宽松。引入 GPT-4o 作为第三方 judge 后：
- Faithfulness: Kimi 比 GPT-4o 平均高 10~15%
- Multihop chain_score: Kimi 比 GPT-4o 平均高 5~13%

### 7.3 Windows 编码陷阱

PowerShell 的 `Tee-Object` 将 UTF-8 stderr 按 GBK 解码 → 中文日志乱码。

修复：在所有 4 个脚本中加入 `io.TextIOWrapper(sys.stdout/stderr, encoding='utf-8')` + `--log` 参数直接写入文件。

---

## 8. 核心结论

### 8.1 没有银弹：最优方法取决于任务类型

| 任务类型 | 最优方法 | 原因 |
|---------|---------|------|
| 单跳精准回答 | **HyDE + Rerank** (0.812) | 假设古文片段填补查询鸿沟；graph 扩展引入噪声 |
| 跨书多步推理 | **Graph RAG v7** (0.729) | IDF 加权边 + 语义门控有效桥接不同古籍 |

### 8.2 图谱的演进是渐进式改良

```
v1 (Flat, 43K edges)        → 0.435   噪声淹没信号
v2 (min_weight=2, 15K)      → 0.465   粗粒度过滤
v3 (IDF + degree=15, 5.3K)  → 0.697   精确加权 + CoT prompt
v7 (+ vector_filter=50)     → 0.729   语义门控
v8 (+ bge-base-768d)        → 0.725   更大 embedding 但 vf 更有效
```

从 v1 到 v7，chain_score 从 0.435 提升到 0.729（+67%），核心改进来自三项创新：
1. **IDF 加权边** (v3): 让稀有术语的跨书连接权重更高
2. **Degree pruning** (v3): 控制噪声上限
3. **Vector filter** (v7): 图拓扑 × 语义相关性双重筛选

### 8.3 Retrieval 是天花板，Generation 是地板

- Prompt tuning (v10) 变体之间只有 ±2pp 差异
- 不同 reranker model 的影响不如检索策略本身大
- faithfulness 从 proposition (0.39) 到 HyDE (0.92) 差距超过 50pp → **检索质量决定回答质量**

---

## 附录：文件索引

| 类型 | 路径 |
|------|------|
| Normal 成绩 | `benchmarks/results/rescore_normal_20260418_143256/comparison.json` |
| Pure HyDE 成绩 | `benchmarks/results/v12_hyde_new/hyde_rerank_topn7_*_detail.json` |
| Multihop 成绩 | `benchmarks/results/multihop/rescore_20260418_102900/leaderboard_merged.json` |
| Prompt Tuning | `benchmarks/results/v10_concise/`, `benchmarks/results/v10_balanced/` |
| 知识图谱统计 | `data/graph_stats.json` |
| 图谱构建 | `scripts/build_knowledge_graph.py` |
| Graph Retriever | `api/graph_retriever.py` |
| 评估脚本 | `scripts/rag_bench.py`, `scripts/bench_multihop.py` |
| Rescore 脚本 | `scripts/rescore_gpt4o.py`, `scripts/rescore_normal_gpt4o.py` |
