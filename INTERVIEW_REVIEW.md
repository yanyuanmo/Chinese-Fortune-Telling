# Chinese Fortune Telling RAG 项目面试复习文档

> **Self-contained 面试复习手册** — 覆盖架构、分块、检索、评估、图 RAG、设计缺陷与改进。  
> 阅读本文档即可完整理解项目，不需要再看代码。

---

## 目录

1. [项目一句话概述](#1-项目一句话概述)
2. [技术栈总览](#2-技术栈总览)
   - 2.1 端到端架构流程图
3. [数据层：三本古籍与分块策略](#3-数据层三本古籍与分块策略)
4. [向量索引层：BGE 嵌入 + ChromaDB](#4-向量索引层bge-嵌入--chromadb)
5. [检索层：5 种检索策略演进（8 版 Graph RAG）](#5-检索层5-种检索策略演进)
   - 5.1 Baseline（纯向量 k=3）
   - 5.2 Hybrid（BM25 + 向量 EnsembleRetriever）
   - 5.3 Rerank（向量 + CrossEncoder）— 含 ms-marco 踩坑故事
   - 5.4 HyDE + BGE Rerank（生产方案）
   - 5.5 Graph RAG（跨书多跳）— 8 版演进概要
6. [生成层：Kimi LLM + 角色扮演 Prompt](#6-生成层kimi-llm--角色扮演-prompt)
7. [评估体系](#7-评估体系)
   - 7.1 RAGAS 单跳评估（22 题）— 含 Evaluator 演变与 Self-eval Bias 修复
   - 7.1.1 RAGAS 四指标详细计算过程
   - 7.2 多跳推理评估（36 题）
8. [Benchmark 数据全表](#8-benchmark-数据全表) — GPT-4o 重评 + 方法演进消融 + Prompt Tuning
9. [知识图谱构建细节](#9-知识图谱构建细节)
10. [Graph RAG 检索全流程详解](#10-graph-rag-检索全流程详解)
    - 10.1 BFS 细节 — 加权 BFS 算法 + hop=1 vs hop=2 A/B 实验
    - 10.2 旧版 interleave 问题（已修复）
    - 10.3 与业界 Graph RAG 方案对比 — 含 Node Degree 分析
    - 10.4 HyDE + Graph 架构不兼容分析（v5 实验）— 含 seed/neighbor 量化
11. [发现的设计缺陷与修复](#11-发现的设计缺陷与修复)
12. [可改进方向汇总](#12-可改进方向汇总)
    - 12.1 已评估但不适用的 Advanced Graph RAG 技术
    - 12.2 最高价值未来方向：Edge Description（边语义描述）
13. [简历 Bullet Points](#13-简历-bullet-points)
14. [面试高频 Q&A](#14-面试高频-qa)
15. [核心代码速查](#15-核心代码速查)
16. [项目文件结构](#16-项目文件结构)
17. [STAR 挑战故事集](#17-star-挑战故事集)（面试行为面必备）

---

## 1. 项目一句话概述

基于 3 本中国古典命理学文献（《三命通会》《滴天髓》《子平真诠》），构建了一套 **RAG（Retrieval-Augmented Generation）系统**，支持用户用自然语言提问命理问题，系统从古籍中检索相关段落并生成回答。项目实现了 **5 种检索策略**的渐进式优化（从纯向量到 HyDE+Rerank 再到 Graph RAG），并建立了完整的 **RAGAS 自动化评估体系** 和 **多跳推理 Benchmark**。

---

## 2. 技术栈总览

| 组件 | 技术选型 | 说明 |
|------|----------|------|
| **Embedding** | `BAAI/bge-small-zh-v1.5` (384d) / `BAAI/bge-base-zh-v1.5` (768d) | 中文优化 Bi-Encoder，v8 升级到 base |
| **Reranker** | `BAAI/bge-reranker-base` | XLMRoberta 架构的 Cross-Encoder (278M) |
| **向量数据库** | ChromaDB | 本地持久化，`chroma_db_bge/` (small) / `chroma_db_bge_base/` (base) |
| **LLM** | Kimi `moonshot-v1-32k` | 主生成模型，通过 OpenAI 兼容 API 调用 |
| **HyDE LLM** | Kimi `moonshot-v1-8k` | 用于生成假设性文档（HyDE 步骤） |
| **框架** | LangChain | 检索链、History-aware Retriever、Stuff Chain |
| **知识图谱** | NetworkX | 668 节点、5,255 条 IDF 加权边（v3 优化后） |
| **后端** | FastAPI | 暴露 REST API |
| **前端** | Streamlit | 聊天界面 + 侧边栏配置 |
| **评估 LLM** | GPT-4o (OpenAI) | 第三方 Judge，消除 self-eval bias |
| **评估 Embedding** | `BAAI/bge-small-zh-v1.5` | 替换英文 all-MiniLM-L6-v2，修复 relevancy 失真 |
| **评估数据** | RAGAS 22 题 + 自定义 chain_score 36 题 | 单跳 + 跨书多跳 |
| **CoT Prompt** | 多跳 Benchmark 专用 | 「各书要点 → 逐步推理 → 结论」结构化推理 |
| **部署** | Docker Compose + Terraform (Azure) | 容器化 API + App |

### 2.1 端到端架构流程图

```
┌─────────────────── Offline Pipeline ───────────────────┐
│                                                         │
│  3 本古籍 PDF                                           │
│      │                                                  │
│      ▼                                                  │
│  build_index_bge.py                                     │
│  ├─ 页眉清理 (regex)                                    │
│  ├─ 按书策略分块 (○ / 章节号 / 空行合并)                  │
│  ├─ secondary_split (>1500字二次切 / <50字合并)           │
│  └─ bge-small-zh 编码 → ChromaDB (668 vectors)          │
│                                                         │
│  build_knowledge_graph.py                               │
│  ├─ 36 个桥接术语 × 倒排索引                             │
│  ├─ IDF 加权：term_idf = log(N/df)                       │
│  ├─ 枚举跨书 chunk pair → edge_weight += IDF             │
│  ├─ min_weight=3 过滤 + max_degree=15 Union 剪枝         │
│  └─ NetworkX 图 (668 nodes, 5255 edges) → graph.pkl     │
└─────────────────────────────────────────────────────────┘

┌─────────────────── Runtime Pipeline ───────────────────┐
│                                                         │
│  Streamlit (fortune_app.py)                             │
│      │  HTTP                                            │
│      ▼                                                  │
│  FastAPI (fortune_main.py)                              │
│      │                                                  │
│      ▼                                                  │
│  fortune_langchain_utils.py ← 生产路径                   │
│  ┌──────────────────────────────────────────────┐       │
│  │ Step 0: create_history_aware_retriever       │       │
│  │         (Kimi-8k 压缩多轮对话为 standalone Q) │       │
│  │ Step 1: HyDE (Kimi-8k 生成仿古文 80-150字)   │       │
│  │ Step 2: ChromaDB similarity_search (k=8/15)  │       │
│  │ Step 3: bge-reranker-base CrossEncoder 精排   │       │
│  └──────────────────────────────────────────────┘       │
│      │  top_n docs                                      │
│      ▼                                                  │
│  create_stuff_documents_chain (Kimi-32k 生成回答)        │
│                                                         │
│  graph_retriever.py ← 仅 Benchmark 路径                 │
│  ┌──────────────────────────────────────────────┐       │
│  │ 向量宽召回 → BFS 图扩展 → CrossEncoder 精排   │       │
│  └──────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

---

## 3. 数据层：三本古籍与分块策略

### 3.1 三本书的基本信息

| 书名 | 分块数 | 分块策略 | 典型 chunk 大小 |
|------|--------|----------|----------------|
| **三命通会** | 419 chunks | 按 `○` 标记切分（298 个 ○ → 二次分割后 419 块） | avg ~530 字 |
| **滴天髓** | 154 chunks | 按空行切分 + 合并（四字诀 + 原注 + 任氏曰 = 一个语义块） | avg ~340 字 |
| **子平真诠** | 95 chunks | 按汉字章节序号切分（如"十八、论四吉神能破格"） | avg ~400 字 |
| **合计** | **668 chunks** | — | — |

### 3.2 分块策略的设计理念

每本书结构不同，所以用**不同的分块函数**：

- **三命通会**：全书有 298 个 `○` 标记，每个 ○ 开头一个语义小节（如「○论五行生成」），用正则前瞻断言 `(?=○)` 切分，保留 ○ 在每个 chunk 开头。切完后有些节太长（>1500 字），走二次分割。
- **滴天髓**：结构是「四字诀 + 原注 + 任氏曰」三段一组。按空行切分后，用合并逻辑把短段（≤100 字的四字诀）和后续的原注、任氏曰合并成一个完整语义块。
- **子平真诠**：按「一、」「二、」…「十八、」等中文数字章节号切分。

### 3.3 二次分割（secondary_split）

```
if chunk > 1500 字:
    RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100,
        separators=["\n\n", "\n", "。", "，", "、", " ", ""]
    )
if chunk < 50 字:
    合并到下一个 chunk（避免孤立碎片）
```

**RecursiveCharacterTextSplitter 工作原理**：按 separators 优先级从高到低尝试分割。先尝试 `\n\n`（段落），不行就 `\n`（行），再不行就 `。`（句号）……一直递归到单字符。这保证了尽量在语义边界处分割。

### 3.4 预处理：页眉清理

PDF 提取后带有页眉（如 `三命通会·9·`、`-9/153-`），用正则全部清除。  
连续 3 行以上空行压缩为 2 行。

---

## 4. 向量索引层：BGE 嵌入 + ChromaDB

- **模型**：`BAAI/bge-small-zh-v1.5`，384 维向量，专为中文优化。
- **归一化**：`encode_kwargs={"normalize_embeddings": True}`，使余弦相似度退化为点积（ChromaDB 默认用余弦距离）。
- **存储**：ChromaDB 本地目录 `chroma_db_bge/`，collection 名 `"langchain"`。
- **总量**：668 个向量（= 668 chunks）。

> **面试追问**：为什么选 `bge-small` 而不是 `bge-base` 或 `bge-large`？
> 
> 答：项目初期为了快速迭代选了 small（模型只有 ~24M 参数，加载快）。Benchmark 显示 small 已经足够（context_recall 能到 0.91+），但 base/large 理论上对古汉语文言文的语义理解更好，是一个可以升级的方向。

---

## 5. 检索层：5 种检索策略演进

### 5.1 Baseline — 纯向量检索

```
query → bge-small 编码 → ChromaDB similarity_search(k=3) → 返回 top 3 documents
```

- 最简单的方案，直接用查询文本的 embedding 去 ChromaDB 检索。
- **问题**：k=3 太少，古汉语语义空间和现代中文有 gap，纯向量召回率不够。

### 5.2 Hybrid — BM25 + 向量混合

```
query → [BM25 检索 top-k] + [向量检索 top-k] → EnsembleRetriever(RRF 融合) → 返回 top-k
```

- **EnsembleRetriever**：LangChain 的混合检索器，内部使用 **Reciprocal Rank Fusion (RRF)** 将两路结果合并。
- **RRF 公式**：`score(doc) = Σ 1/(rank_i + 60)`，对每路结果按排名倒数求和。`weights` 参数控制各路权重比。
- **为什么有效**：BM25 做精确关键词匹配（"正官"、"偏财" 等术语在文言文中是原文出现的），向量做语义匹配（理解"财运好不好" ≈ "正财旺相"）。两路互补。

### 5.3 Rerank — 向量宽召回 + CrossEncoder 精排

```
query → 向量检索(k=fetch_k, 如 15) → CrossEncoder 对每个 (query, doc) pair 打分 → 取 top_n
```

- **CrossEncoder vs Bi-Encoder 区别（面试高频）**：
  - **Bi-Encoder**（bge-small）：query 和 doc 分别编码成两个向量，再算内积/余弦。query 和 doc 之间没有 attention 交互，所以快但不够精准。
  - **CrossEncoder**（bge-reranker-base）：把 `[query, doc]` 拼接成一个序列送入 XLMRoberta，内部做 full cross-attention，输出一个 logit 分数。更精准但更慢（每个 pair 要过一遍完整 Transformer）。
  
- **两者通常是同一模型家族（如 BAAI/bge），但不是同一个模型**。bge-small 是 Bi-Encoder，bge-reranker-base 是 Cross-Encoder，参数不同、训练目标不同。

> **踩坑故事（面试加分项）**：最初使用的 CrossEncoder 是 `cross-encoder/ms-marco-MiniLM-L-6-v2`（英文 MSMARCO 数据集训练的模型）。结果 Rerank 后效果反而变差——因为这个模型完全不理解古汉语，反而**打乱了 EnsembleRetriever 已经合理的 RRF 排序**。配置文件 `configs/rag/hybrid_rerank_bge.yaml` 中有明确记录。后来切换到 `BAAI/bge-reranker-base`（中文训练的 XLMRoberta CrossEncoder）才解决。
>
> **启示**：Cross-Encoder 必须和语料语言匹配。英文 Reranker 处理中文/古汉语 = 随机打分。

### 5.4 HyDE + BGE Rerank（生产方案 ★）

这是最终部署到生产的检索策略。注意 **History-aware Retriever 是这条管道的最外层**，整个流程分 4 步：

```
            用户多轮对话 + 最新问题
                    │
            ┌───────▼─────────────┐
            │  Step 0: History-    │  LLM 把多轮对话 + 最新问题压缩成
            │  aware Condense      │  独立的 standalone query
            │ (Kimi-8k)           │  例：上文聊正官格 → 用户问"那偏财呢？"
            │                     │  → condensed: "偏财格的论述是什么？"
            └───────┬─────────────┘
                    │ standalone_query
            ┌───────▼────────┐
            │  Step 1: HyDE   │  LLM 生成 80-150 字仿古文假设性文档
            │ (Kimi-8k)       │  "你是命理学专家，仿照古典文言文风格写一段..."
            └───────┬────────┘
                    │ hypothetical_doc
            ┌───────▼────────┐
            │  Step 2: 宽召回  │  用 hypothetical_doc 的 embedding 去 ChromaDB 检索
            │ similarity_search│  k=15（Benchmark 最优值；生产代码写了 k=8，是已知 bug）
            │ (k=15)          │
            └───────┬────────┘
                    │ 15 candidates
            ┌───────▼────────┐
            │  Step 3: 精排    │  用 *standalone_query*（不是 hypothetical_doc）
            │ CrossEncoder     │  对每个 (query, candidate) 打分
            │ top_n=7         │  取分数最高的 7 个
            └───────┬────────┘
                    │ 7 documents → 送入 LLM 生成回答
```

**Step 0 的实现**：LangChain 的 `create_history_aware_retriever(llm, hyde_rerank_retriever, contextualize_prompt)` 把 HyDE+Rerank retriever 包在里面。它先用 LLM + `contextualize_prompt` 把 `(chat_history, latest_question)` 压缩成 standalone query，再交给内层的 HyDE+Rerank。如果没有历史对话（首轮提问），则直接透传原始问题。

**HyDE 的关键洞察**：用户问题是现代汉语（"正官格怎么看？"），但文献是文言文（"正官佐君之臣......"）。二者的 embedding 有 gap。HyDE 让 LLM 先"猜"一段古文应该长什么样，用这段**假设性文档的 embedding** 去检索——embedding 空间对齐了。

**为什么限定 80-150 字？** 太短（<80字）→ 关键词密度不足，embedding 信息量有限，检索召回率低。太长（>150字）→ LLM 容易跑偏到与问题无关的内容，生成的假文本引入噪音，反而污染 embedding。80-150 字是实验找到的 sweet spot，刚好覆盖 2-3 个核心术语 + 关联论述。

**为什么 Step 3 用原始问题而不是 hypothetical_doc？** 因为 CrossEncoder 需要判断的是"这个 doc 能不能回答用户的问题"，而不是"这个 doc 和 LLM 写的假文有多像"。

### 5.5 Graph RAG（跨书多跳 ★）— 8个版本演进

专门为**跨书推理**设计的检索方案。完整流程详见 [第 10 节](#10-graph-rag-检索全流程详解)。

**版本演进概要**（面试高频 ★）：

| 版本 | 核心改动 | Multihop (GPT-4o) | 面试要点 |
|------|---------|:---:|------|
| v1 | Flat graph, 43K edges | 0.435 | 首次引入知识图谱，但噪声边太多 |
| v2 | min_weight=2, 15K edges | 0.465 | 粗过滤，cross_book_hit 69%→89% |
| v3 | **IDF 加权 + degree=15 剪枝** | 0.546 / 0.697(+CoT) | ★ 核心创新：稀有术语高权重 |
| v4 | 升级 reranker v2-m3 (568M) | 0.637 | ★ 负面实验：大模型不一定好 |
| v5 | HyDE + Graph | 0.671 | ★ 负面实验：架构不兼容 |
| v6 | max_neighbors 消融 (30→10) | 0.692 | 邻居太少会崩溃 (mn15=0.208) |
| v7 | **vector_filter_k=50** | **0.729** | ★ 核心创新：拓扑×语义双重门控 |
| v8 | bge-base-zh (768d) + k=15 | 0.725 | 更大 embedding，稳定提升 |

> **面试话术**："8 个版本中有 2 个核心创新（v3 IDF 加权、v7 语义门控）和 2 个有价值的负面实验（v4 大 reranker、v5 HyDE+Graph）。v1→v7 chain_score 从 0.435 提升到 0.729（+67%），其中 prompt（CoT）贡献最大。"

---

## 6. 生成层：Kimi LLM + 角色扮演 Prompt

- **生产 Prompt**：系统设定为「果赖」(Guo Lai) 这个传统命理师角色，带角色扮演口吻回答。
- **Benchmark Prompt**：不带角色扮演，纯中文直接作答（`bench_qa_prompt`），因为角色扮演会降低 RAGAS 的 `answer_relevancy` 分数（废话和套话被判定为不相关）。
- **Stuff Chain**：检索到的 top_n 个文档直接拼接塞进 prompt 的 `{context}` 占位符（LangChain 的 `create_stuff_documents_chain`），然后交给 LLM 生成最终回答。

---

## 7. 评估体系

### 7.1 RAGAS 单跳评估（22 题）

**QA 数据集生成**：用 Kimi 从 ChromaDB 的 chunk 中自动生成 QA，精选 22 道代表性单跳问题，每题有 `question`、`ground_truth`（黄金答案）、`contexts`（黄金上下文 chunk）。

> **单跳题目示例**：  
> **Q**: 在传统命理中，阳刃格使用官杀时有哪些特殊考量？  
> **Golden Answer**: 阳刃格使用官杀有特殊考量：与他格不同，阳刃用杀是为制刃，故喜财印，忌制伏。若阳刃用官，透刃不虑，因官能制刃；但阳刃露杀，透刃则无功，因刃能合杀，使杀贪合忘克……  
> **来源**: 《子平真诠》单本  
> **特点**: 只需从单本书中检索 1-2 个 chunk 即可完整回答，无需跨书推理。

**4 个 RAGAS 核心指标**：

| 指标 | 衡量什么 | 计算方式（简化） |
|------|----------|-----------------|
| **faithfulness** | 回答是否忠实于检索到的上下文 | LLM 把回答拆成若干 claim → 逐条检查是否被 context 支持 → 支持率 |
| **answer_relevancy** | 回答是否切题 | LLM 从回答反向生成问题 → 与原始问题的 embedding 相似度 |
| **context_recall** | 检索到的 context 是否覆盖了黄金答案的要点 | LLM 把黄金答案拆成要点 → 检查每个要点是否被 context 覆盖 |
| **context_precision** | 检索到的 context 中有用内容的比例 | 有用 chunk 排名越靠前得分越高（@K precision 风格） |

**评估者 (Judge) 演变（面试重点 ★）**：

| 阶段 | 评估 LLM | 评估 Embedding | 问题 |
|------|---------|---------------|------|
| v1-v8 初始 | **Kimi (self-eval)** | all-MiniLM-L6-v2 (**英文**) | 自评偏宽松 +10~33%；英文 embedding 测中文 relevancy 失真 |
| 最终修复 | **GPT-4o** | **bge-small-zh-v1.5 (中文)** | 两个问题都解决 |

> **英文 Eval Embedding 对 relevancy 的影响**：  
> RAGAS 的 answer_relevancy 指标内部使用 embedding cosine similarity 衡量 answer 和 question 的语义距离。用英文 all-MiniLM-L6-v2 评估中文内容 → 系统性低估 relevancy（差 ~12.5pp）。换成中文 bge-small-zh-v1.5 后 relevancy 从 0.524 → 0.656。**这个 bug 我们直到对比实验时才发现。**

> **⚠️ Self-eval Bias 修复（面试话术）**：  
> 初始版本 RAG 生成用 Kimi，评估 Judge 也用 Kimi——「既当运动员又当裁判」。引入 GPT-4o 作为第三方 Judge 后：faithfulness 平均被扣 10~15pp，multihop chain_score 被扣 5~13pp。**这说明初始评估结果系统性偏乐观**，GPT-4o 对推理链中间步骤要求更严格。

#### RAGAS 四指标详细计算过程（面试必考）

**① Faithfulness（忠实度）** — 回答是否有依据？

```
输入: answer, contexts

Step 1: LLM 抽取 claims
   answer = "正官格喜身旺，忌伤官破格" 
   → claims = ["正官格喜身旺", "正官格忌伤官破格"]

Step 2: LLM 逐条验证 claim vs context
   claim "正官格喜身旺" → context 中有 "正官佐君之臣……身旺方能任之" → ✅ supported
   claim "正官格忌伤官破格" → context 中有 "伤官见之则破" → ✅ supported

Step 3: faithfulness = supported_claims / total_claims = 2/2 = 1.0
```

> **面试追问**：为什么 faithfulness 和 hallucination 有关？  
> 答：如果 LLM 编造了一个 context 中没有的 claim（幻觉），该 claim 会被判 unsupported → faithfulness 下降。所以 faithfulness 本质上是 **anti-hallucination 指标**。

**② Answer Relevancy（回答相关度）** — 回答是否切题？

```
输入: question, answer

Step 1: LLM 从 answer 反向生成 N 个问题（默认 N=3）
   answer = "正官格喜身旺，忌伤官破格" 
   → generated_q1 = "正官格有什么喜忌？"
   → generated_q2 = "什么情况下正官格会被破？"
   → generated_q3 = "正官格对身旺有何要求？"

Step 2: 分别计算 generated_q 和 original_question 的 embedding 余弦相似度
   sim(original_q, gen_q1) = 0.85
   sim(original_q, gen_q2) = 0.72
   sim(original_q, gen_q3) = 0.80

Step 3: answer_relevancy = mean(similarities) = (0.85 + 0.72 + 0.80) / 3 = 0.79
```

> **为什么本项目 relevancy 总是偏低（~0.5-0.6）？**  
> (1) 角色扮演 Prompt 让回答包含「善哉」「天机如此」等无关内容 → 反向生成的问题偏离原意  
> (2) 古汉语+白话混合回答的 embedding 质量不稳定  
> (3) RAGAS 默认用 OpenAI embedding 计算相似度，对中文不够友好

**③ Context Recall（上下文召回率）** — 检索到的内容覆盖了多少黄金答案？

```
输入: ground_truth, contexts

Step 1: LLM 把 ground_truth 拆成若干 claim（要点）
   ground_truth = "正官格需身旺方能任之。若伤官见之则破。月令正官，不可再见七煞混杂。"
   → claims = ["正官格需身旺", "伤官破正官格", "月令正官忌七煞混杂"]

Step 2: LLM 逐条检查每个 claim 是否被 contexts 支持
   "正官格需身旺" → context chunk 1 包含 → ✅ attributed
   "伤官破正官格" → context chunk 2 包含 → ✅ attributed  
   "月令正官忌七煞混杂" → contexts 中未找到 → ❌ not attributed

Step 3: context_recall = attributed / total = 2/3 = 0.667
```

> **关键**：context_recall 用的是 **ground_truth**（黄金答案）而不是 answer（模型回答）。它衡量的是"检索系统有没有把该找的内容都找到"，和 LLM 的生成质量无关。

**④ Context Precision（上下文精确度）** — 检索到的内容中有用的排前面吗？

```
输入: question, ground_truth, contexts (有序列表)

Step 1: LLM 对每个 context chunk 判断：这个 chunk 对回答问题有用吗？
   context[0] = "正官佐君之臣……身旺方能任之" → ✅ relevant (rank=1)
   context[1] = "论五行生克……"              → ❌ irrelevant (rank=2)
   context[2] = "伤官见官，百祸其端"          → ✅ relevant (rank=3)

Step 2: 计算 Precision@K 的加权平均
   P@1 = 1/1 = 1.0    (rank 1 relevant)
   P@2 = 1/2 = 0.5    (rank 2 irrelevant, 不参与计算)
   P@3 = 2/3 = 0.667  (rank 3 relevant)
   
   context_precision = (P@1 + P@3) / 2 = (1.0 + 0.667) / 2 = 0.833
   （只对 relevant 位置求 P@K，再取平均）
```

> **公式**：$\text{context\_precision} = \frac{\sum_{k=1}^{K} (\text{Precision@k} \times \text{rel}_k)}{\text{total relevant}}$  
> 其中 $\text{rel}_k = 1$ 当第 k 个 chunk 有用，$\text{Precision@k} = \frac{\text{relevant in top-k}}{k}$  
> **直觉**：有用的 chunk 排在前面 → precision 高；有用的 chunk 被噪音 chunk 挤到后面 → precision 低。

**RAGAS Judge LLM**：最终使用 GPT-4o 作为评分 LLM（第三方评估，避免 self-eval bias）。评估 Embedding 使用 bge-small-zh-v1.5（中文），避免英文 embedding 导致的 relevancy 失真。

### 7.2 多跳推理评估（36 题）

**数据集生成流程**：

```
3 本书 → 3 个书对 (三命×滴天, 三命×子平, 滴天×子平)
↓
对每个书对：
  1. 从 chunk_index 中找共享 bridge_terms 的跨书 chunk pair
  2. 发 24 个 pair 给 Kimi（MULTIHOP_PROMPT），要求设计多跳推理题
  3. ~50% 返回 null（两段文确实凑不成多跳推理）
  4. 保留约 12 道有效题
↓
合计 36 道题
```

**每道题的结构**（以真实题目为例）：
```json
{
  "question": "根据《三命通会》中关于正财的论述和《子平真诠》中关于星辰与格局的讨论，如何理解在八字命理中正财与格局的关系？",
  "reasoning_chain": [
    "步骤1: 《三命通会》中提到正财喜旺食丰盈，日主刚强力可胜，若财多身弱则平生破败事无成。",
    "步骤2: 《子平真诠》中强调八字格局以月令配四柱，星辰好坏不能决定格局成败，格局既成即使满盘孤辰入煞也不损其贵。",
    "步骤3: 因此，正财的吉凶不仅取决于其本身旺衰，更与日主强弱和整体格局有关。"
  ],
  "required_hops": 3,
  "golden_answer": "在八字命理中，正财的吉凶不仅取决于其本身的旺衰，更与日主的强弱和整体格局的配合密切相关……",
  "source_chunks": ["《三命通会》正财篇原文", "《子平真诠》格局篇原文"],
  "metadata": {"book1": "三命通会", "book2": "子平真诠", "bridge_term": "正财"}
}
```

> **与单跳的核心区别**：  
> - 问题**显式涉及两本书**（如"根据《三命通会》和《子平真诠》"），需要检索到两本书的相关段落  
> - `reasoning_chain` 定义了必须跨书串联的推理步骤，`bridge_term`（如"正财"）是连接两本书的术语枢纽  
> - 这正是 Graph RAG 的核心场景：通过知识图谱的跨书边发现这些桥接关系

**评估指标**：

| 指标 | 定义 |
|------|------|
| **chain_score** | 把推理链的每个步骤和模型回答给 Kimi 评分（1/0），`chain_score = 覆盖步骤数 / 总步骤数` |
| **hop_ok_rate** | `chain_score ≥ 0.6` 的题目占比（"基本推对了"） |
| **full_ok_rate** | `chain_score = 1.0` 的题目占比（"完全推对"） |
| **cross_book_hit** | 检索结果是否覆盖了两个源书籍（Graph RAG 的核心指标） |

---

## 8. Benchmark 数据全表

> **所有数据均使用 GPT-4o 评估 + 中文 bge-small-zh-v1.5 Eval Embedding**。  
> 完整数据和方法描述见 [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md)，本节聚焦面试叙事。

### 8.1 RAGAS 单跳评估（22 题）— GPT-4o + Chinese Eval Embedding

| # | 配置 | 类型 | Faithfulness | Relevancy | Recall | Precision | **AVG** |
|:-:|------|------|:---:|:---:|:---:|:---:|:---:|
| 1 | **HyDE + Rerank** (topn7) | HyDE | **0.917** | 0.651 | 0.829 | 0.850 | **0.812** |
| 2 | Graph RAG v8 (bge-base k15) | Graph | 0.841 | **0.656** | **0.849** | **0.868** | 0.804 |
| 3 | Graph RAG v7 (vf50) | Graph | 0.902 | 0.639 | 0.794 | 0.833 | 0.792 |
| 4 | Hybrid (BM25+向量) | Baseline | 0.762 | 0.668 | 0.846 | 0.885 | 0.790 |
| 5 | Graph RAG v5 (HyDE+Graph) | 混合 | 0.861 | 0.622 | 0.784 | 0.847 | 0.779 |
| 6 | Graph RAG v3 (CoT) | Graph | 0.852 | 0.620 | 0.775 | 0.854 | 0.775 |
| 7 | Hybrid + BGE Rerank | Baseline | 0.755 | 0.664 | 0.720 | 0.894 | 0.758 |
| 8 | Proposition (向量) | Proposition | 0.492 | 0.667 | 0.567 | 0.719 | 0.611 |
| 9 | Proposition (混合) | Proposition | 0.392 | 0.628 | 0.438 | 0.453 | 0.478 |

**关键发现（面试话术★）**：

1. **HyDE + Rerank 是单跳最优**：faithfulness 0.917（最高）、AVG 0.812。HyDE 的假设古文填补了 query-document 词汇鸿沟。
2. **Graph RAG v8 紧随其后（AVG 0.804）**：得益于 bge-base-zh (768d) 更强的语义表示能力。
3. **Hybrid 无 rerank 出人意料地排 #4**（AVG 0.790）：BM25 对文言文术语的字面匹配 + 向量语义匹配互补，无需 reranker 也有竞争力。
4. **Proposition Indexing 全面失败**：古文命题拆分质量差，上下文碎片化导致 faithfulness 暴跌至 0.39~0.49。
5. **answer_relevancy 普遍偏低**（0.52~0.67）：角色扮演 Prompt 的套话 + 古汉语混合回答的 embedding 质量限制了这个指标。

### 8.2 多跳推理评估（36 题）— GPT-4o Rescore

| # | 配置 | Chain Score | Hop OK | Full OK | Cross Hit |
|:-:|------|:---:|:---:|:---:|:---:|
| 1 | **v7_vf50** (语义门控) ★ | **0.729** | 72.2% | 47.2% | 77.8% |
| 2 | v8_bge_base_k15 | 0.725 | **77.8%** | 44.4% | 88.9% |
| 3 | v8_bge_base_k20 | 0.725 | **77.8%** | 44.4% | **91.7%** |
| 4 | v8_hyde_bge_base | 0.720 | 75.0% | 41.7% | 27.8% |
| 5 | v7_vf100 | 0.720 | 69.4% | **50.0%** | 86.1% |
| 6 | v3_cot | 0.697 | 69.4% | 44.4% | 88.9% |
| 7 | v8_reranker_large | 0.692 | 72.2% | 41.7% | 80.6% |
| 8 | v6_mn10 | 0.692 | 69.4% | 36.1% | 91.7% |
| 9 | v5_hyde_cot | 0.671 | 72.2% | 41.7% | 83.3% |
| 10 | v4_cot_v2m3 | 0.637 | 63.9% | 41.7% | 88.9% |
| — | *以下为非 CoT 配置或基线* | | | | |
| 12 | hyde_no_graph | 0.593 | 58.3% | 38.9% | 38.9% |
| 14 | v3_idf (无 CoT) | 0.546 | 58.3% | 25.0% | 83.3% |
| 16 | v2_flat | 0.465 | 44.4% | 27.8% | 88.9% |
| 17 | hybrid_feb | 0.458 | 44.4% | 11.1% | 100.0% |
| 18 | graph_rag_v1 | 0.435 | 47.2% | 13.9% | 69.4% |

**关键发现（面试话术★）**：

1. **v7 vector_filter_k=50 是多跳冠军（0.729）**：图拓扑 × 语义相关性双重筛选是关键创新。
2. **v8 bge-base 紧随其后（0.725）**：更大 embedding 模型稳定提升 ~3pp，但不如 vector_filter 有效。
3. **v1→v7 的演进：chain_score 从 0.435 到 0.729（+67%）**：核心改进来自 IDF 加权 + degree pruning + vector_filter。
4. **大模型 reranker 反而更差**：v2-m3 (568M)=0.637 < v4 reranker-large (560M)=0.692 < base (278M)=0.725。多语言通用模型对古文排序不如中文专精小模型。
5. **hybrid 的 cross_book_hit=100% 是偶然**：BM25 靠关键词匹配碰巧命中多书内容，但 chain_score 仅 0.458（倒数第三）。
6. **HyDE+Graph (v5)=0.671 弱于纯 Graph (v3_cot)=0.697**：架构不兼容，详见第 10.4 节。

### 8.3 方法演进消融分析（面试核心叙事 ★★★）

```
                              GPT-4o chain_score
v1 Flat Graph (43K edges)        0.435   ← 噪声淹没信号
v2 min_weight=2 (15K edges)      0.465   ← 粗粒度过滤 (+6.9%)
v3 IDF + degree=15 (5.3K)        0.546   ← 精确加权，无 CoT
  └─ + CoT prompt                0.697   ← prompt 巨幅提升 (+27.7%)
v4 升级 reranker v2-m3           0.637   ← 反而更差 (−8.6%)  
v5 HyDE + Graph                  0.671   ← 架构不兼容 (−3.7%)
v6 max_neighbors=10              0.692   ← 减少邻居噪声 (−0.7%)  
v7 vector_filter_k=50            0.729   ← 语义门控 (+4.6%) ★ BEST
v8 bge-base-zh (768d)            0.725   ← 更大 embedding (−0.5% vs v7, +4.0% vs v3_cot)
```

**三大核心主线**（面试必答）：

1. **图谱质量线**（v1→v3）：边从 43K 降到 5.3K，IDF 加权让稀有术语的跨书连接更有价值。chain_score +25.5%。
2. **生成质量线**（v3→v3_cot）：CoT prompt「各书要点 → 逐步推理 → 结论」让 LLM 显式整合跨书内容。chain_score +27.7%——**prompt 工程的 ROI 远超模型升级**。
3. **检索精度线**（v3_cot→v7）：vector_filter 给 BFS 加了语义门控——邻居必须同时出现在 top-50 向量结果中才被保留。chain_score +4.6%。

**负面实验同样有价值**：
- 升级 reranker 到 v2-m3 → 更差（−8.6%）。说明古文领域中文专精小模型 > 多语言大模型。
- HyDE + Graph → 更差（−3.7%）。说明两种检索策略解决的是同一问题的不同侧面，叠加会稀释各自优势。

### 8.4 Prompt Tuning 天花板实验

在最优检索配置 (graph_rag_v8_bge_k15) 上测试 3 种 prompt 风格：

| Prompt 风格 | Faithfulness | Relevancy | Recall | Precision |
|------------|:---:|:---:|:---:|:---:|
| **default** | **0.841** | **0.656** | **0.849** | **0.868** |
| concise (150-300字) | 0.826 | 0.550 | 0.836 | 0.871 |
| balanced (200-400字+引用) | 0.821 | 0.520 | 0.834 | 0.860 |

**结论**：三种 prompt 之间差异 ≤ 2pp，**prompt tuning 已触顶**。生成质量的天花板由检索质量决定——检索给了什么，LLM 就只能在这个范围内回答。

---

## 9. 知识图谱构建细节

### 9.1 构建流程

```
668 chunks
    ↓ 对每个 chunk 做桥接术语匹配
    ↓ BRIDGE_TERMS = 36 个命理学核心术语
    ↓ （"官星","财星","印绶","伤官","格局","用神","日主","五行",...）
    ↓
建立倒排索引：term → [chunk_ids]
    ↓
计算每个术语的 IDF 权重：
    term_idf[term] = log(N / df)     # N=668, df=包含该术语的 chunk 数
    # 高频术语（「五行」df=158）→ IDF≈1.4，低贡献
    # 低频术语（「调候」df=12）→ IDF≈3.9，高贡献
    ↓
枚举跨书 chunk pair（对每个 term，两两配对共享该 term 的 chunk）：
    edge_weight[(A,B)] += term_idf[term]    # IDF 加权累计
    edge_terms[(A,B)].append(term)
    ↓
过滤 + 度限制剪枝：
    1. min_weight=3：只保留 IDF 累计权重 ≥3 的边
    2. max_degree=15（Union 策略）：每个节点只保留权重最高的 top-15 条边
       Union 策略：如果 A 或 B 任一端点想保留这条边，就保留
       （比 Intersection 策略更温和，避免过度剪枝）
```

**IDF 加权 vs 旧版简单计数（v2→v3 改进）**：

| 方案 | 边权重定义 | 边数 | 平均度 | 问题 |
|------|-----------|------|--------|------|
| 旧版 (v2) | 共享术语种类数 | 15,698 | 47.0 | 高频术语（五行等）制造大量噪音边 |
| 新版 (v3) | Σ IDF(term_i) | **5,255** | **15.7** | IDF 降权高频术语 + 度限制剪枝 |

> **注**：代码中用 `set(info["bridge_terms"])` 对每个 chunk 内部去重——即使一个 chunk 里「正官」出现了 5 次，也只算匹配 1 次。权重反映的是**术语种类的 IDF 加权总和**。

### 9.2 图统计

| 属性 | v2（旧版） | v3（IDF+剪枝） |
|------|-----------|---------------|
| 节点数 | 668 | 668 |
| 边数 | 15,698 | **5,255** |
| 平均度 | 47.0 | **15.7** |
| 边权重范围 | min=2, max=9 | min=3.0, max=16.5 |
| 频率最高的桥接术语 | 伤官(185)、印绶(184) | 同（术语集不变） |

### 9.3 设计选择与权衡

- **无向图（`nx.Graph()`）**：代码中用的是 NetworkX 无向图，而非有向图 `DiGraph`。原因：边的语义是"两个 chunk 共同讨论了某组术语"，这是一个**对称关系**（A 和 B 共享「正官」=  B 和 A 共享「正官」），没有方向性。如果改成有向图，需要定义方向的语义（如"A 解释了 B 的概念" vs "B 引用了 A 的论述"），但我们的术语共现方法无法区分方向。

  > **面试追问**：什么情况下需要有向图？  
  > 答：(1) 有因果/引用关系时（如论文引用图：A→B 表示 A 引用 B）；(2) 有时序依赖时（如章节顺序：第 3 章基于第 2 章的概念）；(3) 实体关系图中关系本身有方向（如「正官→日主：辅佐」）。本项目的术语共现是无方向的，用无向图是正确的选择。

- **边权重 = 共享术语的 IDF 加权总和（v3 改进后）**：权重反映两个 chunk 之间概念重叠的"信息量"。共享低频术语（如「调候」IDF≈3.9）的边权重远高于共享高频术语（如「五行」IDF≈1.4）的边。这比旧版的简单计数更合理——两个 chunk 都因为提到「五行」而连边没什么信息量，但都讨论「调候」很可能真的相关。

  > **面试追问**：为什么用 IDF 而不用 TF-IDF？  
  > 答：因为每个 chunk 内部已经去重（`set(bridge_terms)`），TF 对所有 term 都是 1，TF-IDF 退化为 IDF。而且我们关注的是"术语跨多少文档出现"（DF），不是在单个 chunk 中的频率。

  > **面试追问**：权重在检索时有用到吗？  
  > 答：**有**（v3 改进后）。BFS 遍历邻居时会按边权重降序排序，优先走高权重边（高 IDF 的术语共现 → 更可能相关）。加上 `max_neighbors=30` 截断，相当于只走最相关的邻居，而非随机遍历。

- **min_weight=3**（v3 改进后用 IDF 加权阈值 3.0）：只保留 IDF 累计权重 ≥3 的跨书边。比旧版的 min_weight=2（简单计数）过滤更精准——两个 chunk 共享 2 个高频术语（如「五行」+「官星」, IDF≈1.4+1.4=2.8 < 3）不会建边，但共享 1 个低频术语+1个中频术语（如「调候」+「印绶」, IDF≈3.9+1.3=5.2 > 3）会建边。
- **36 个桥接术语是硬编码的**：由开发者手工选取命理学核心概念。优点：可控、精准。缺点：覆盖不全（如"禄神""桃花"没有收录），且无法自动扩展。改进方向：用 TF-IDF 或 PMI 自动发现高区分度的跨书术语。
- **只建跨书边（核心设计决策）**：同书 chunk 之间不建边。原因有四：
  1. **同书 chunk 已被向量检索覆盖**：同书相关 chunk 在 embedding 空间中天然距离近，向量宽召回已经能召回它们。图邻居名额花在同书上等于重复向量检索的工作。
  2. **图的唯一价值是补跨书关联**：多跳问题的难点在于答案散落在不同书里，而不同书的表述风格差异大、embedding 距离远，向量检索很难跨书召回。图的边就是专门弥补这个 gap。
  3. **max_degree 预算有限**：每个节点只保留 top-15 条边。如果允许同书边，同书内高频共现术语（如「日主」「五行」出现在同一本书 419 个 chunk 中的几十个里）会产生海量同书边，挤占跨书边名额，直接损害跨书召回率。
  4. **同书边的 IDF 信号弱**：同一本书讨论的术语高度重叠（主题连贯），同书 chunk 对共享大量低 IDF 通用术语，边权虚高但语义关联弱。跨书 chunk 对共享的术语则更具判别力——两本不同风格的书同时讨论某术语，说明该术语是真正的语义桥梁。

  > **面试话术**："图是向量检索的补充而非替代，它只解决向量检索做不好的事——跨书关联。同书边不仅没用，还会稀释有限的 max_degree 名额。"

---

## 10. Graph RAG 检索全流程详解

Graph RAG 是本项目中最复杂的检索策略，分 4 步：

```
用户问题: "正官格在大运中如何变化？"

═══ Step 1: 向量宽召回 seed docs ═══
    similarity_search(query, k=10) → 10 个 seed documents
    这些可能全来自《三命通会》（因为三命有 419/668 = 63% 的 chunk）

═══ Step 2: 知识图谱 BFS 邻居扩展 ═══
    对每个 seed chunk_id，在知识图谱中做 BFS（hop=1）：
    
    seed_A (三命通会, 包含"正官","格局")
        → 邻居1 (子平真诠, 共享"正官","格局")  ✓ 异书，加入
        → 邻居2 (滴天髓, 共享"官星","大运")    ✓ 异书，加入
        → 邻居3 (三命通会, 共享"正官")          ✗ 同书，跳过
    
    每个 seed 最多扩展 max_neighbors=30 个异书邻居。
    去重后得到 neighbor_docs。

═══ Step 3: 合并 ═══
    all_docs = seed_docs + neighbor_docs
    典型数量：10 seeds + 20~30 neighbors ≈ 30~40 个候选

═══ Step 4: CrossEncoder 精排 + 多样性保证 ═══
    对 ALL candidates 用 bge-reranker-base 打分（不截断！全量精排）：
    
    for each (query, doc) pair:
        score = XLMRoberta([query, doc]).logit
    
    # tokenizer 参数: max_length=256, truncation=True, padding=True
    # 即 query + doc 合计最多 256 tokens，超出部分被截断。
    # 对于平均 ~400 字的 chunk，256 tokens 大约覆盖 170-200 个汉字。
    
    ranked = sort by score, descending
    
    _diverse_top_n(ranked, top_n=7)：
        取前 7 个
        如果 7 个全来自同一本书 → 把第 7 个替换为排名最高的异书 doc
        保证最终结果至少覆盖 2 本书
```

**`_diverse_top_n` 实际代码**（`graph_retriever.py`）：

```python
def _diverse_top_n(ranked: list[tuple], top_n: int) -> list:
    selected = [doc for _, doc in ranked[:top_n]]
    books_in = {(doc.metadata or {}).get("book", "") for doc in selected}
    if len(books_in) >= 2:
        return selected  # 已经有多样性，直接返回

    # 找候选中分数最高的、不在 selected 中的异书 doc
    selected_set = set(id(d) for d in selected)
    for _, doc in ranked[top_n:]:
        b = (doc.metadata or {}).get("book", "")
        if b not in books_in:
            selected[-1] = doc  # 把最低分的同书 doc 换掉
            break
    return selected
```

> **面试追问**：为什么只替换最后一个而不是插入？  
> 答：保持 `top_n` 不变。如果只需保证至少 2 本书，只换掉分数最低的那个同书 doc 即可，代价最小。

### 10.1 BFS 细节

#### 加权 BFS 算法步骤（`_bfs_neighbors`）

```python
def _bfs_neighbors(self, seed_id: str) -> list[str]:
    seed_book = chunk_index[seed_id]["book"]
    visited = {seed_id}
    frontier = [seed_id]
    neighbors = []

    for _ in range(self.hop):              # hop=1，只执行一轮
        next_frontier = []
        for node in frontier:
            # ★ 关键：按边权重降序排列邻居
            adj = [(nbr, graph[node][nbr]["weight"])
                   for nbr in graph.neighbors(node)
                   if nbr not in visited]
            adj.sort(key=lambda x: x[1], reverse=True)

            for nbr, _w in adj:
                visited.add(nbr)
                nbr_book = chunk_index[nbr]["book"]
                if nbr_book != seed_book:       # 只取异书邻居
                    neighbors.append(nbr)
                    if len(neighbors) >= self.max_neighbors:
                        return neighbors
                next_frontier.append(nbr)  # 同书节点也加入frontier（为多hop准备）
        frontier = next_frontier
    return neighbors
```

**核心设计点**：

1. **按边权重降序遍历**（v3 新增）：`adj.sort(key=weight, reverse=True)` 确保优先走高 IDF 共现的邻居。当 `max_neighbors=30` 截断时，留下的是与 seed 术语重叠信息量最大的跨书 chunk，而非随机遍历到的。
2. **只取异书邻居**：`nbr_book != seed_book` 过滤。seed 已经是同书内容了，图扩展的唯一目的是从其他书中补充关联内容。
3. **同书节点仍加入 frontier**：即使同书邻居不会进入结果，它们仍被加入 `next_frontier`——这样在 hop>1 时，BFS 可以"借道"同书节点跳到更远的异书节点。
4. **max_neighbors=30**：硬上限防止高 degree seed 导致候选池爆炸。实测中 10 个 seed 平均各扩展 ~10 个邻居，总候选池约 30-40 个。

#### 为什么只用 1-hop？（A/B 实验数据）

做了 v3_idf_hop1 vs v3_idf（hop=2）的严格 A/B 对比（36 题多跳测试集）：

| | hop=1 | hop=2 | delta |
|---|:---:|:---:|:---:|
| chain_score | 0.6412 | 0.6412 | **±0.0** |
| cross_book_hit | **88.9%** | 83.3% | **−5.6pp** |
| 逐题对比 | — | better=3, worse=4, same=29 | — |

hop=2 分数完全持平，跨书命中率反而下降。原因：

1. **"多跳问题" ≠ "图上多跳"**：多跳问题的意思是答案散落在 2 本书里，但在图上 1 跳就够了——seed 在 book A，沿一条跨书边走 1 hop 就到了 book B。**1 graph-hop = 1 book-crossing**，已经覆盖了所有多跳题的跨书需求。
2. **2-hop 噪声远大于信息增益**：1-hop 邻居最多 15 个（max_degree=15），都与 seed 直接共享术语。2-hop 是邻居的邻居，潜在候选 15×15=225 个，但它们只是碰巧和 seed 的邻居共享某个术语，与原始 query 的语义关联大幅衰减。
3. **2-hop 路径可以绕回同书**：A(三命)→B(子平)→C(三命) 是合法的 2-hop 路径，这些「出去又绕回来」的同书 chunk 通过 reranker 可能被排到前面，挤掉真正需要的异书 chunk——这正是 cross_book_hit 下降 5.6pp 的原因。

> **面试话术**："实测 hop=2 chain_score 不变但 cross_book_hit 下降 5.6pp。因为答案只散落在 2 本书中，1 hop 已经完成跨书，第二跳引入的是邻居的邻居——语义关联弱、还可能绕回同书，是纯噪声。"

### 10.2 旧版的 interleave 问题（已修复）

**旧版**在 Step 4 之前有一个 `_interleave_by_book()` 步骤：
```python
# 旧代码
interleaved = _interleave_by_book(all_docs)     # 按书轮转排列
candidates = interleaved[:top_n * 4]             # 取前 28 个进 CrossEncoder
```

**问题**：`similarity_search()` 返回的 `List[Document]` 没有分数，BFS 邻居也没有分数。合并后的 all_docs 是无分数的 → interleave 按书轮转 → `[:28]` 截断 → 一些高相关的 seed 可能因为书轮转的位置问题被截掉，而低相关的邻居反而进了 CrossEncoder 窗口。

**修复**：直接把 all_docs 全量（~40 条）送进 CrossEncoder，不做截断。40 条对 CrossEncoder 来说计算量完全可以接受。让 CrossEncoder 的分数做唯一的排序依据。

### 10.3 与业界 Graph RAG 方案对比（面试重点）

#### 10.3.1 三种主流 Graph RAG 范式

| 维度 | Microsoft GraphRAG | LlamaIndex KG | 本项目 |
|------|-------------------|---------------|--------|
| **节点类型** | 实体（人、概念、事件） | 实体 + 关系三元组 | **文本 chunk** |
| **边类型** | LLM 提取的语义关系 | LLM 提取的三元组 | **共享桥接术语的跨书 chunk 对** |
| **图构建** | LLM 自动抽取（贵、慢、不确定） | LLM 三元组抽取 | **规则匹配（41 术语 × 倒排索引），零 LLM 调用** |
| **检索方式** | 社区检测 → 分层摘要 → 摘要检索 | 子图检索 + 三元组上下文 | **向量宽召回 → BFS 邻居 → CrossEncoder 精排** |
| **社区/聚类** | Leiden 社区检测 + 层次摘要 | 无 | **无** |
| **适合场景** | 开放域大规模文档、摘要型问答 | 需要精确实体关系的 QA | **小规模多文档跨书推理** |
| **构建成本** | 高（每个 chunk 过 LLM 抽取） | 中-高 | **极低（纯 Python 正则 + Counter）** |

#### 10.3.2 为什么不用 Microsoft GraphRAG？

1. **语料太小**：只有 668 chunks / 3 本书。Microsoft GraphRAG 的社区检测 + 层次摘要是为数万文档设计的，668 个节点做 Leiden 社区检测意义不大。
2. **古汉语实体抽取不靠谱**：LLM 对文言文的 NER 能力远不如现代文本。让 LLM 从「官星佐君之臣……」中提取实体和关系，准确率不可控。
3. **核心需求是跨书关联**：我们不需要知道「正官」和「日主」之间的语义关系是什么，只需要知道「哪些 chunk 讨论了同一组概念」→ 术语 co-occurrence 就足够了。
4. **成本考虑**：GraphRAG 对 668 chunks 做实体抽取 ≈ 668 次 LLM 调用，而本方案 0 次。

**面试话术**："我们的需求是跨文档多跳检索，不是开放域摘要。在 668 chunks 的小语料上，术语共现图 + BFS + CrossEncoder 精排已经把 cross_book_hit 从 44% 拉到 89%。引入 GraphRAG 的复杂度不 justified。"

#### 10.3.3 本方案的核心设计决策

| 决策 | 我们的选择 | 通常做法 | 为什么 |
|------|-----------|---------|--------|
| 节点粒度 | chunk 级别 | 实体级别 | 古汉语 NER 不靠谱，chunk 级别免抽取 |
| 边的定义 | 术语共现 | LLM 语义关系 | 确定性强、零成本、可复现 |
| 图遍历 | BFS hop=1 | 多跳/随机游走/PPR | avg_degree=47 太高，hop>1 爆炸 |
| 排序 | CrossEncoder 精排全量候选 | 图距离 / embedding 相似度 | CrossEncoder 是最精准的排序器 |
| 只建跨书边 | ✓ | 通常全连 | 我们的目标就是跨书推理 |
| 多样性保证 | `_diverse_top_n` 替换 | MMR / 社区多样性 | 简单有效，1 个替换就够 |

#### 10.3.4 已解决：Node Degree 过大（v2→v3）

本方案原来最大的结构性问题是 **平均度 47** — 远高于典型知识图谱（通常 avg_degree < 10）。**v3 版本通过 IDF 加权 + 度限制剪枝已将其降至 15.7。**

**旧版为什么度这么大？**

```
668 chunks × 36 术语 → 术语覆盖面很高
例：「五行」出现在 158 个 chunk 中 → 跨书配对数 = C(n_A, 1) × C(n_B, 1) 级别
    热门术语「伤官」出现在 185 个 chunk → 跨书 pair 数百甚至上千
多个术语的 pair 集合叠加 → 边数膨胀到 15,698
avg_degree = 2 × 15698 / 668 ≈ 47
```

**v3 修复方案（已实施）**：

1. **IDF 加权边权重** ✅：
   - `edge_weight = Σ IDF(term_i)`，其中 `IDF(term) = log(N/df)`, N=668
   - 「五行」IDF≈1.4（太常见）→ 贡献的权重小；「调候」IDF≈3.9（只在少数 chunk 出现）→ 贡献的权重大
   - 配合 `min_weight=3` 阈值，自动过滤仅靠高频术语连接的噪音边

2. **度限制剪枝（Union 策略）** ✅：
   - 对每个节点，只保留权重最高的 top-15 条边
   - Union 策略：如果边的**任一端点**想保留它（都在各自的 top-15 中），就保留
   - 比 Intersection（双方都在 top-15 才保留）更温和，避免信息丢失

3. **BFS 按权重降序遍历** ✅：
   - 邻居按边权重排序后再遍历，截断时保留最相关的

**效果**：

| 指标 | v2（旧版） | v3（修复后） |
|------|-----------|-------------|
| 边数 | 15,698 | 5,255（-66.5%） |
| 平均度 | 47.0 | 15.7（-66.6%） |
| hop_ok≥0.6 | 69.4% | 77.8%（+8.3pp） |
| chain_score | 0.644 | 0.641（持平） |

chain_score 持平的原因：检索质量提升但 LLM 生成是瓶颈。加了 CoT prompt 后 chain_score 才飙升至 0.787。

**未实施的备选方案**（留作面试讨论素材）：

- **分层术语**：核心术语（出现<50次）× 2 权重，高频术语（出现>150次）× 0.5 权重
- **动态 max_neighbors**：根据 seed 度数动态调整
- **Personalized PageRank**：从 seed 出发跑 PPR，按相关性自然衰减

### 10.4 HyDE + Graph 架构不兼容分析（v5 实验）— 含 seed/neighbor 量化

#### 10.4.1 v5 方案设计

在 v5 中，我们尝试将 HyDE 和 Graph RAG 结合：用 HyDE 生成的假设文档做向量检索获取 seed，再走 Graph BFS 展开邻居，最后 CrossEncoder 精排。理论上，HyDE 能提升 seed 质量 → 更好的图展开 → 更好的最终结果。

```
v5 流程: question → HyDE生成 → similarity_search(hyde_text, k=10) → BFS邻居 → CrossEncoder精排(original_query) → top_n=7
对比纯HyDE: question → HyDE生成 → similarity_search(hyde_text, k=15) → CrossEncoder精排 → top_n=5
对比v3 Graph: question → similarity_search(query, k=10) → BFS邻居 → CrossEncoder精排 → top_n=7
```

#### 10.4.2 实验结果：v5 全面劣于各自单独方案

**多跳数据集（36 题，GPT-4o chain_score）**：

| 配置 | chain_score | hop_ok | full_ok |
|------|-----------|--------|--------|
| **v7 vf50** ★ | **0.729** | 72.2% | **47.2%** |
| **v3 Graph + CoT** | 0.697 | 69.4% | 44.4% |
| v5 HyDE+Graph+CoT | 0.671 | 72.2% | 41.7% |
| 差值 (v5 vs v3) | **-3.7pp** | +2.8pp | -2.7pp |

**单跳数据集（22 题，GPT-4o RAGAS）**：

| 配置 | Faithfulness | Relevancy | Recall | Precision | AVG |
|------|:---:|:---:|:---:|:---:|:---:|
| **纯 HyDE+Rerank** ★ | **0.917** | **0.651** | **0.829** | **0.850** | **0.812** |
| Graph v3 (无HyDE) | 0.852 | 0.620 | 0.775 | 0.854 | 0.775 |
| v5 HyDE+Graph | 0.861 | 0.622 | 0.784 | 0.847 | 0.779 |

v5 在单跳上仅与 v3 持平（AVG 0.779 vs 0.775），远不及纯 HyDE（0.812）。在多跳上 chain_score 也低于 v3 CoT 和 v7。

#### 10.4.3 根因分析：Graph Expansion 稀释了 HyDE 的改善

**实验 1：文档重叠率分析**

对 22 道单跳题，比较三个方案最终返回的文档 ID 集合：

| 比较 | 重叠率 | 含义 |
|------|--------|------|
| v3 ∩ 纯HyDE | 0/154 (0.0%) | Graph 和 HyDE 产出的文档**完全不同** |
| v5 ∩ 纯HyDE | 1/154 (0.6%) | HyDE 加入 Graph 后，结果和纯 HyDE **几乎无关** |
| v5 ∩ v3 | 100/154 (64.9%) | v5 的结果和 v3（无HyDE的Graph）**高度重合** |

**结论**：Graph BFS 展开 + 重排后，HyDE 改善 seed 质量的效果被**完全覆盖**。无论 seed 是 HyDE 检索还是原始 query 检索，经过 Graph 展开后最终结果趋同。

**实验 2：seed vs neighbor 量化分析（面试重点 ★）**

对 20 道题，跟踪每个最终 top-7 文档来自 seed 还是 graph neighbor：

```
 Q  seeds  nbrs total  S_in7 N_in7        top7组成    seed均分    nbr均分       
 1     10   126   136      5      2      SSSNNSS      0.91    -2.62
 4     10   110   120      4      3      SNNSNSS      2.75     0.23
 9     10   122   132      1      6      SNNNNNN     -3.74    -4.33
10     10    48    58      7      0      SSSSSSS      2.55    -1.94
14     10    64    74      7      0      SSSSSSS      2.18    -1.95
                          ...（共 20 题）
```

**汇总统计**：

| 指标 | 值 | 说明 |
|------|-----|------|
| seed 占候选池 | 10/112 = **8.9%** | 10 个 seed 被稀释在 ~112 个候选中 |
| seed 占 top-7 | 76/140 = **54.3%** | 被 reranker 放大了 **6 倍** |
| neighbor 占 top-7 | 64/140 = **45.7%** | 仍有近半结果是 graph neighbor |
| seed 平均 reranker 分 | **-0.206** | — |
| neighbor 平均 reranker 分 | **-2.477** | — |
| 差值 (seed - neighbor) | **+2.271** | seed 的相关性显著高于 neighbor |

#### 10.4.4 核心洞察：「简单精排问题」vs「困难精排问题」（面试话术 ★）

**纯 HyDE 流程**：vector search 直接返回 **15 个语义最相关候选** → reranker 从 15 个高质量候选中选 5 个 → 任务简单，效果好。

**Graph 流程**：vector search 返回 10 个 seed → BFS 展开 ~102 个 neighbor → 共 ~112 个候选 → reranker 从 112 个混合质量候选中选 7 个 → 任务困难。

> **面试话术**："Graph expansion 用拓扑关联替代了语义关联，把候选池从 10 个高质量 seed 稀释成 112 个混合候选。Reranker（278M 的 bge-reranker-base）能力有限，虽然成功将 seed 从 8.9% 放大到 54.3%（6 倍），但仍有 45.7% 的 neighbor 噪音渗入 top-7。本质上，graph expansion 把一个『15 选 5 的简单精排问题』变成了『112 选 7 的困难精排问题』。"

> **追问：graph 不准？reranker 不准？**
>
> 更精确的说法：
> - **Graph** 的问题不是"不准"，而是**目标不对**——它的边代表术语共现（拓扑关联），不代表查询相关性。neighbor 与 seed 主题相关，但与用户问题未必相关。
> - **Reranker** 不是"不准"，而是**能力有限**——278M 模型面对 112 个候选，区分力不够。102 个 neighbor 里总有几个与 query 有术语表面重合，reranker 给出的分数不够低，于是挤进了 top-7。
> - **根本原因**是架构层面的信息损失：HyDE 改善了 seed 质量，但 graph 展开后 seed 被淹没在 10 倍数量的 neighbor 中。

#### 10.4.5 为什么 HyDE 单独好用但与 Graph 不兼容？

HyDE 在纯向量检索管道中表现优异（context_recall 0.545→0.911），因为：
1. HyDE 生成的假文档与古籍 embedding 对齐 → 向量检索直接返回高质量候选
2. Reranker 只需从 15 个已经不错的候选中精选 5 个 → 容错率高
3. 即使 HyDE 生成了错误内容（"幻觉"），reranker 也能过滤掉（因为用原始 query 精排）

但加入 Graph 后，HyDE 的改善被架构性地抵消：
1. HyDE 改善的 seed（10 个）被 graph 展开为 ~112 个候选
2. 展开后 seed 仅占 8.9% 的候选池 → HyDE 的贡献被稀释到 <10%
3. 最终结果由 graph 拓扑结构主导，而非语义相关性

**最终结论**：HyDE 和 Graph RAG 不是简单的加法关系。它们解决的是**同一个问题的不同侧面**（语义对齐 vs 跨文档关联），但在当前架构下叠加使用会导致信息损失。正确做法是**路由策略**：单跳用 HyDE，多跳用 Graph+CoT。

---

## 11. 发现的设计缺陷与修复

### 缺陷 1: similarity_search 丢弃分数

**问题**：整个代码库中，ChromaDB 的 `similarity_search_with_score()` 从未被调用过（grep 确认 0 处），所有地方都用的 `similarity_search()` 只返回 Document 对象没有分数。这意味着：

- 向量检索的排名信息只体现在返回顺序上（第 1 个 doc 大概比第 10 个好），但具体差多少不知道。
- 无法在合并 seed + neighbors 时做加权排序。

**修复方案**：改用 `similarity_search_with_score()` 获取距离分，在 seed 和 neighbor 合并后、进 CrossEncoder 之前，可以用向量分数做粗排再截断（如果候选数真的很多的话）。但目前因为候选总数只有 ~40，直接全量 CrossEncoder 更干脆。

### 缺陷 2: 生产参数未同步 Benchmark 最优值 ✅ 已修复

**问题**：`fortune_langchain_utils.py` 中生产代码写死了 `hyde_k=8, top_n=5`，但 Benchmark 调参发现最优是 `k=15, top_n=7`。

**修复**：已同步为 `hyde_k=15, top_n=7`。

### 缺陷 3: Graph RAG 未接入生产 API

**问题**：`fortune_main.py` 只调用 `get_fortune_chain()`（HyDE+Rerank），Graph RAG 的 `GraphRetriever` 只在 benchmark 脚本中使用，没有暴露为 API endpoint。用户无法在前端选择 Graph RAG 模式。

### 缺陷 4: Graph BFS hop=1 与多跳题目 required_hops=3 的能力错配 — 已验证 hop=1 是最优

**问题**：Graph RAG 在知识图谱上只做 hop=1 的 BFS，但多跳评估数据集中每道题 `required_hops=3`。

**验证结果**：在 v3 IDF 图（avg_degree=15.7）上实测 hop=1 vs hop=2：
- hop=1: chain_score=0.641, hop_ok=77.8%
- hop=2: chain_score=0.641, hop_ok=75.0%（略差）
- 逐题对比：8 题好 / 9 题差 / 19 题持平 → 随机噪音，无系统优势

**结论**：hop=2 即使在稀疏化后的 IDF 图上仍然引入太多噪音（第二跳候选 ~200+），淹没了 reranker 的精排能力。hop=1 + 高质量邻居（IDF 权重排序）是当前的最优平衡。

**真正的瓶颈**：不在检索端，而在生成端。加上 CoT prompt 后，hop=1 的 chain_score 从 0.641 飙升至 0.787，说明 LLM 只需要正确的引导就能整合跨书内容。

---

## 12. 可改进方向汇总

| # | 方向 | 难度 | 影响 | 状态 | 说明 |
|---|------|------|------|------|------|
| 1 | 生产参数同步 | ★☆☆ | 高 | ✅ 已完成 | `hyde_k=8→15, top_n=5→7` |
| 2 | IDF 加权 + 度限制剪枝 | ★★☆ | 高 | ✅ 已完成 | 边数 -66.5%，hop_ok +8.3pp → chain +25.5% |
| 3 | BFS 按权重排序 | ★☆☆ | 中 | ✅ 已完成 | 优先走高 IDF 邻居 |
| 4 | CoT Prompt | ★☆☆ | **极高** | ✅ 已完成 | chain_score **+27.7%**（GPT-4o 口径） |
| 5 | 429 API 错误重试 | ★☆☆ | 中 | ✅ 已完成 | 3 次指数退避重试 |
| 6 | chain_score>1.0 评分 Bug 修复 | ★☆☆ | 低 | ✅ 已修复 | 截断 LLM 返回的多余 step scores |
| 7 | vector_filter_k (语义门控) | ★★☆ | **高** | ✅ 已完成 | v7 核心创新，chain +4.6% |
| 8 | 升级 embedding bge-small→bge-base | ★★☆ | 中 | ✅ 已完成 | v8, Normal AVG +3pp, Multihop +3pp |
| 9 | GPT-4o 第三方评估 | ★☆☆ | **高** | ✅ 已完成 | 消除 self-eval bias，严格 10~33% |
| 10 | 中文 Eval Embedding | ★☆☆ | 中 | ✅ 已完成 | relevancy +12.5pp（英文→中文） |
| 11 | ~~升级 bge-reranker-v2-m3~~ | ★★☆ | **负面** | ❌ 已验证 | chain_score -13.8% vs base |
| 11b | ~~HyDE + Graph 组合 (v5)~~ | ★★☆ | **负面** | ❌ 已验证 | 架构不兼容，详见 10.4 |
| 12 | Prompt Tuning 天花板 | ★☆☆ | **天花板** | ✅ 已验证 | 3 种 prompt 差异 ≤2pp，检索是瓶颈 |
| 13 | Graph RAG 接入生产 API | ★★☆ | 高 | ⏳ 待做 | 路由策略：单跳 HyDE / 多跳 Graph |
| 14 | 自动化桥接术语发现 | ★★★ | 中 | ⏳ 待做 | 用 TF-IDF/PMI 替代人工 41 个词 |
| 15 | HyDE 缓存 | ★★☆ | 中 | ⏳ 待做 | SQLiteCache |
| 16 | 流式输出 | ★★☆ | 高 | ⏳ 待做 | SSE 流式返回 |
| 17 | 测试集数据泄漏 | ★★★ | 高 | ⏳ 待做 | hold-out chunk 或外部专家题目 |
| 18 | Edge Description（边语义描述） | ★★★ | **高** | ⏳ 待做 | 详见 12.1 |

### 12.1 已评估但不适用的 Advanced Graph RAG 技术

面试中可能被追问"为什么不做 X"，以下逐项说明：

| 技术 | 结论 | 原因 |
|------|------|------|
| Edge Type Filtering | ❌ 不适用 | 图中只有一种边（术语共现跨书边），无类型可选 |
| Similarity Thresholding (cosine > θ) | ✅ 已有等价 | `vector_filter_k=50` 是 rank-based 语义门控，比硬编码阈值更优——不需调超参，rank 天然自适应 |
| Text-ified Graph Triples | ❌ 不适用 | 节点是 text chunk、边是 IDF 数值，不是 `(实体, 关系, 实体)` 三元组；chunk 本身就是文本 |
| Cross-Encoder Rerank | ✅ 已实现 | `bge-reranker-base` 对 ~40 候选全池精排 → top-7 |
| Hybrid Scoring (α×S_semantic + β×W_topology) | ⚠️ 理论可行但风险高 | 引入 α, β 两个超参数，36 道测试集**过拟合风险极高**；reranker 已隐式平衡 seed:neighbor ≈ 50:50 |
| Context Packing | ❌ 无需 | top-7 × ~500 字 ≈ 3,500 字，32K 窗口仅用 ~10%，完全不是瓶颈 |

**面试话术**："我们评估了这些方向——vector_filter_k 已是 similarity thresholding 的等价实现；只有一种边类型和无结构化三元组，edge type filtering 和 triple textualization 没有适用场景；context packing 在 7 chunks / 32K 窗口下不是瓶颈。唯一值得尝试的是 hybrid scoring，但 36 道题不足以调 α/β。"

### 12.2 最高价值未来方向：Edge Description（边语义描述）→ Query-Aware BFS

#### 问题诊断

当前 BFS 遍历是 **query-agnostic** 的——只看静态 IDF 权重，不知道当前问题在问什么：

```
当前:  seed → BFS(按静态 IDF 排序) → vector_filter_k 过滤 → rerank → top-7
                  ↑ 不感知 query
```

`vector_filter_k=50` 在 BFS **之后**做语义过滤，但过滤对象是 neighbor chunk 本身，不是"A→B 这条边的关系是否与 query 相关"。这意味着 BFS 可能走了一条 IDF 很高但与当前问题无关的边，浪费了 `max_neighbors=30` 的有限预算。

#### 目标架构

```
目标:  seed → BFS(按 α×IDF_norm + (1-α)×edge_relevance 排序) → vf_k 过滤 → rerank → top-7
                              ↑
                  cosine(query_emb, edge_desc_emb)   ← 动态、感知 query
```

#### Phase 1: 构建阶段 — 生成边描述并预编码

在 `build_knowledge_graph.py` 建图完成后追加两步：

```python
# ── Step A: LLM 生成边描述 ──────────────────────────────────────────────
def generate_edge_descriptions(G, chunk_index, model="gpt-4o-mini"):
    """为图中每条边生成一句话描述，存入边属性 'description'。"""
    from openai import OpenAI
    client = OpenAI()
    PROMPT = (
        "以下是两段中国古典命理文献的摘录。请用一句话（30-60字）概括"
        "它们之间的关联：为什么这两段可以互相参照？"
        "只输出概括，不要解释。\n\n"
        "【{book_a}】{text_a}\n\n【{book_b}】{text_b}"
    )
    for i, (a, b, data) in enumerate(G.edges(data=True)):
        info_a, info_b = chunk_index[a], chunk_index[b]
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT.format(
                book_a=info_a["book"], text_a=info_a["text"][:300],
                book_b=info_b["book"], text_b=info_b["text"][:300],
            )}],
            max_tokens=100,
        )
        data["description"] = resp.choices[0].message.content.strip()
```

成本估算：5,255 边 × (~300 input + 60 output tokens/边) ≈ **$1–3**（GPT-4o-mini 价格）。

```python
# ── Step B: 预编码边描述为向量 ──────────────────────────────────────────
def encode_edge_descriptions(G, model_name="BAAI/bge-small-zh-v1.5"):
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(model_name)
    descs, keys = [], []
    for a, b, data in G.edges(data=True):
        if "description" in data:
            descs.append(data["description"])
            keys.append((a, b))
    embeddings = encoder.encode(descs, batch_size=64, show_progress_bar=True)
    for (a, b), emb in zip(keys, embeddings):
        G[a][b]["desc_emb"] = emb.tolist()       # list 形式存 pickle
```

同时预计算全局 IDF 范围用于归一化（IDF 范围 ~3–12，cosine 范围 0–1，不归一化会被 IDF 主导）：

```python
weights = [d["weight"] for _, _, d in G.edges(data=True)]
G.graph["weight_min"] = min(weights)
G.graph["weight_max"] = max(weights)
```

#### Phase 2: 检索阶段 — `_bfs_neighbors` 改为 query-aware

当前代码（`graph_retriever.py` L126-129）：

```python
# 现有：仅按静态 IDF 排序
adj = [(nbr, self.graph[node][nbr].get("weight", 0))
       for nbr in self.graph.neighbors(node) if nbr not in visited]
adj.sort(key=lambda x: x[1], reverse=True)
```

改为 **query-aware 混合排序**：

```python
def _bfs_neighbors(self, seed_id: str, query_emb=None) -> list[str]:
    """Query-aware 加权 BFS。
    排序键 = α × norm(IDF) + (1-α) × cosine(query_emb, edge_desc_emb)
    """
    ...
    w_min = self.graph.graph.get("weight_min", 0)
    w_max = self.graph.graph.get("weight_max", 1)
    alpha = 0.5          # IDF 与 query-relevance 的混合权重

    for _ in range(self.hop):
        next_frontier = []
        for node in frontier:
            adj = []
            for nbr in self.graph.neighbors(node):
                if nbr in visited:
                    continue
                edge = self.graph[node][nbr]
                idf_w = edge.get("weight", 0)
                # ── 归一化 IDF 到 [0,1] ──
                idf_norm = (idf_w - w_min) / (w_max - w_min) if w_max > w_min else 0

                # ── query-aware: 边描述与 query 的语义相关度 ──
                edge_rel = 0.0
                if query_emb is not None and "desc_emb" in edge:
                    edge_rel = _cosine(query_emb, edge["desc_emb"])

                score = alpha * idf_norm + (1 - alpha) * edge_rel
                adj.append((nbr, score))

            adj.sort(key=lambda x: x[1], reverse=True)
            ...  # 其余逻辑（visited / 异书过滤 / max_neighbors）不变
```

#### Phase 3: 调用链改动（最小侵入）

在 `_get_relevant_documents` 的 Step 2 之前加一行，复用 vectorstore 已有的 embedding 函数：

```python
# 零额外依赖，embedding 只算一次（~1ms）
query_emb = self.vectorstore._embedding_function.embed_query(query)

# Step 2 遍历时传入
for nbr in self._bfs_neighbors(sid, query_emb=query_emb):
    ...
```

`vector_filter_k` 白名单、reranker、`_diverse_top_n` **全部不变**。

#### 设计决策与面试亮点

| 决策 | 理由 |
|------|------|
| 混合分 `α×IDF + (1-α)×cosine` 而非纯 cosine | 保留拓扑信号，纯语义排序退化为 vector retrieval（失去图的独特价值） |
| 边描述离线生成 | 在线检索零 LLM 调用开销，BFS 中只有 cosine 计算（384d × ~15 邻居 = 微秒级） |
| IDF 做 min-max 归一化 | IDF (3–12) vs cosine (0–1) 量级差 10 倍，不归一化 α 参数无意义 |
| `α=0.5` 起步 | 可 grid search `α ∈ {0.3, 0.5, 0.7}` 做 A/B，但 36 题测试集调参风险大 |
| 只改 BFS adj 排序层 | 下游组件（vf_k、reranker、diverse_top_n）全部不变，最小侵入 |

#### 与现有 `vector_filter_k` 的关系

两者目标相同（过滤掉与 query 无关的拓扑邻居）但切入点不同：

| | vector_filter_k (现有) | Query-Aware BFS (新) |
|---|---|---|
| **作用时机** | BFS 之后，硬过滤 | BFS 之中，软排序 |
| **判断对象** | neighbor chunk 本身 | 两端 chunk 的关系描述 |
| **信息粒度** | "这段文本是否与 query 语义相关" | "这条边连接的原因是否与 query 相关" |
| **互补性** | ✅ 可叠加使用 | ✅ BFS 先选好的走，vf_k 再筛一遍 |

最优方案是**两者叠加**：query-aware BFS 让 `max_neighbors=30` 的预算分配到最相关的边上，`vector_filter_k` 在此基础上再做一次语义硬过滤。

**面试话术**："如果继续优化，最值得做的是给边加语义描述——让 BFS 从 query-agnostic 变成 query-aware。实现方式是离线用 LLM 为 5,255 条边生成关系描述并预编码为向量，在线 BFS 时用 `α×IDF_norm + (1-α)×cosine(query, edge_desc)` 做混合排序。这和 vector_filter_k 互补——前者在 BFS 内部做软排序优化预算分配，后者在 BFS 之后做硬过滤。改动只影响 BFS 排序层，下游 reranker 和多样性组件完全不变。"

### 关于数据泄漏（面试重点）

QA 数据集的 `question` 和 `ground_truth` 都是从 ChromaDB 中的 chunk 生成的。评估时 RAG 从同一个 ChromaDB 检索——这意味着"黄金答案对应的那段 chunk"一定在检索库里，且和问题高度相关。这会人为抬高 `context_recall` 和 `context_precision`。

**更好的做法**：hold-out 一部分 chunk 不入库，或使用外部专家手写的 QA。

---

## 13. 简历 Bullet Points

以下是可以直接放简历的量化成果（所有数据基于 GPT-4o 第三方评估）：

> - Built a RAG system over 3 classical Chinese divination texts (668 chunks), implementing **9 retrieval strategies** with automated evaluation (RAGAS + custom chain_score)
> - Achieved **faithfulness 0.917** and **AVG 0.812** with HyDE + BGE Cross-Encoder Rerank pipeline, evaluated by **GPT-4o as independent judge** with Chinese eval embedding
> - Constructed an **IDF-weighted knowledge graph (668 nodes, 5,255 edges)** for cross-document reasoning, evolving through **8 versions** from 0.435 to **0.729 chain_score (+67%)**
> - Designed **vector_filter_k** — a novel topology × semantics dual-gating mechanism for graph retrieval — achieving **#1 on multihop benchmark** (chain_score 0.729, full_ok 47.2%)
> - Conducted **2 high-value negative experiments**: (1) larger reranker (568M) underperformed base (278M) by −13.8%; (2) HyDE+Graph combination showed architectural incompatibility via quantitative seed/neighbor analysis (seed=8.9% pool → 54.3% top-7)
> - Identified and fixed **self-eval bias** (Kimi judge inflated 10–33%) and **English eval embedding** bug (relevancy +12.5pp after switching to Chinese embedding)

---

## 14. 面试高频 Q&A

### Q1: 什么是 RAG？为什么需要 RAG？

**A**: RAG = Retrieval-Augmented Generation。LLM 的训练数据有截止日期，且不包含你的私有文档。RAG 在生成前先从外部知识库检索相关文档，塞到 prompt 的 context 里，让 LLM "看着回答"。好处：减少幻觉（有出处）、无需微调（换文档就行）、可解释（能追溯到哪段原文）。

### Q2: Bi-Encoder 和 Cross-Encoder 有什么区别？

**A**: 
- **Bi-Encoder**：query 和 doc 分别编码成向量，用内积/余弦算相似度。优点：doc 向量可以预计算存数据库，查询时只需编码 query → O(1) 检索（用 ANN）。缺点：query 和 doc 之间没有 cross-attention，表达能力有限。
- **Cross-Encoder**：把 [query; SEP; doc] 拼成一个序列过 Transformer，有 full cross-attention。优点：更精准。缺点：每个 (query, doc) pair 都要过一遍模型，不能预计算 → O(N) 复杂度。
- **实践中组合使用**：先用 Bi-Encoder 从数据库里粗筛 top-k（宽召回），再用 Cross-Encoder 对 k 个候选精排（精排）。本项目就是这么做的。

### Q3: 什么是 HyDE？为什么有效？

**A**: HyDE = Hypothetical Document Embedding。让 LLM 先根据问题"写"一段假的回答（hypothetical document），用这段假文本的 embedding 去检索。

为什么有效：用户问题和文献之间有语言 gap（现代中文 vs 文言文）。HyDE 让 LLM 把问题"翻译"成和文献风格相似的文本，使 embedding 对齐。就像你搜论文时，先写一段想象中论文应该长什么样，再用这段文本去搜，效果比用问题直接搜好。

本项目数据：HyDE 使单跳 AVG 从 0.790（hybrid baseline）提升到 0.812，faithfulness 从 0.762 到 0.917。但在多跳场景中，HyDE + Graph 组合反而更差（0.671 vs v3_cot 0.697），因为 graph expansion 把 HyDE 改善的 seed 质量稀释了。

### Q4: EnsembleRetriever 的 RRF 是怎么回事？

**A**: Reciprocal Rank Fusion。多路检索器各自返回排名列表，RRF 把排名转化为分数进行融合：

```
RRF_score(doc) = Σ_i  weight_i / (rank_i + k)
```

其中 k=60 是常量，防止排名第 1 的分数过大。这样即使一个 doc 在某路检索中排名很低，只要在另一路排名高，融合后仍有机会胜出。`weights` 参数调节各路的重要性（如 BM25:向量 = 0.4:0.6）。

### Q5: 你的知识图谱和 Microsoft GraphRAG 有什么区别？

**A**: Microsoft GraphRAG 用 LLM 自动提取实体和关系，构建一个**实体关系图**，然后做社区检测、层次聚类摘要。我的知识图谱更简单直接：

- **节点 = 文本 chunk**（不是实体）
- **边 = 共享命理术语的跨书 chunk 对**（不是 LLM 提取的关系）
- **不做社区检测或摘要**，只用 BFS 找邻居

优点：构建完全不依赖 LLM，确定性强、成本低、可复现。
缺点：粒度粗（chunk 级别而非实体级别），边的语义质量依赖人工选取的 36 个桥接术语。

### Q6: RAGAS 的 context_recall 怎么算的？

**A**: RAGAS 让 Judge LLM 把 `ground_truth`（黄金答案）拆成若干 claim（要点），然后逐条检查：这个 claim 是否被检索到的 context 内容支持。

```
context_recall = 被 context 支持的 claim 数 / 总 claim 数
```

所以它衡量的是"你检索到的内容，覆盖了标准答案多大比例的要点"。高 recall 意味着关键信息都被捞到了。

### Q7: chain_score 怎么算？和 RAGAS 有什么区别？

**A**: chain_score 是我们自定义的多跳推理指标，用于评估 Graph RAG。

做法：把标准推理链（如 3 个步骤）和模型回答一起给 Judge LLM（最终使用 GPT-4o），让它逐步打分（0 / 0.5 / 1）。`chain_score = 覆盖步骤数 / 总步骤数`。

和 RAGAS 的区别：
- RAGAS 评的是单跳 QA："回答对不对、检索全不全"
- chain_score 评的是推理链完整性："3 步推理你覆盖了几步"
- chain_score 更适合评估跨文档多跳能力

### Q8: 为什么 answer_relevancy 总是偏低？

**A**: 两个原因：
1. **角色扮演 Prompt**：生产 Prompt 让 LLM 扮演"果赖"命理师，回答中包含"善哉""天机如此"之类的角色扮演套话。RAGAS 的 relevancy 会从回答反向生成问题，套话导致反向问题跑偏 → 和原始问题的 embedding 相似度低。
2. **古汉语 + 中文的 embedding 对齐问题**：RAGAS 的 relevancy 用 embedding 相似度，对于混合文言文/白话的回答，embedding 质量本身就有不确定性。

### Q9: 都有哪些延迟瓶颈？

**A**: HyDE + BGE Rerank 的 p50 延迟 = 18.6s，拆解如下：

| 步骤 | 耗时 | 说明 |
|------|------|------|
| HyDE 生成 | ~3-5s | Kimi API 调用生成假设文档 |
| ChromaDB 检索 | <0.5s | 本地向量检索很快 |
| BGE CrossEncoder | ~1-2s | 15 个 pair 的 XLMRoberta 推理，CPU |
| Kimi 生成回答 | ~10-12s | 主要瓶颈，受 API 限速和模型推理速度影响 |

优化方向：HyDE 缓存（避免重复 query）、CrossEncoder 改 GPU（本项目 CPU 就够了）、流式输出（不减延迟但体感好）。

### Q10: 你做了哪些改进？效果如何？

**A**: 分三轮改进，检索图优化→生成端 prompt→检索精度提升：

**第一轮：知识图谱 IDF 优化（v1→v3）**
- IDF 加权边权重：用 `log(N/df)` 替代简单计数，降低高频术语（「五行」等）对边权重的贡献
- 度限制剪枝：Union 策略保留 top-15 条边，边数从 43K→5.3K
- BFS 按权重排序：优先探索高 IDF 邻居
- 结果：chain_score 从 0.435(v1) → 0.546(v3)，+25.5%

**第二轮：CoT Prompt 工程（v3→v3_cot）**
- 要求 LLM 按"各书要点 → 逐步推理 → 结论"结构作答
- 结果：chain_score **+27.7%**（0.546→0.697，GPT-4o 口径）

**第三轮：语义门控（v7）**
- vector_filter_k=50：BFS 邻居必须同时出现在 top-50 向量检索结果中
- 结果：chain_score **+4.6%**（0.697→0.729），多跳 #1

**关键教训（面试核心话术★）**：小规模专业领域 RAG 中，检索质量达到一定水平后，**prompt 工程的 ROI 远超模型升级**。我们还验证了升级 reranker（bge-reranker-v2-m3，568M）反而降低了 −13.8% chain_score，因为多语言通用模型在古典中文领域的排序不如中文专精小模型。**没有银弹**：单跳最优是 HyDE+Rerank（0.812），多跳最优是 Graph v7（0.729）——正确做法是路由策略。

### Q11: 你试过把 HyDE 和 Graph RAG 结合吗？效果如何？

**A**: 试了，效果反而更差。这是一个有价值的**负面实验（面试加分项）**。

**数据**（GPT-4o 评估）：
- 多跳 chain_score：v5 HyDE+Graph = 0.671 < v7 vf50 = 0.729（-8.0pp）
- 单跳 AVG：v5 = 0.779 < 纯 HyDE = 0.812（-3.3pp）

**根因**：Graph expansion 把 HyDE 改善的 10 个高质量 seed 稀释成 ~112 个混合候选。量化分析显示：seed 仅占候选池 8.9%，但 reranker 将其放大到 top-7 的 54.3%（6 倍放大）。这说明 reranker 在努力过滤噪音，但 278M 模型面对 112 个候选区分力有限，仍有 45.7% 的 graph neighbor 渗入最终结果。

**核心洞察**：纯 HyDE 是"15 选 5 的简单精排"，Graph 把它变成了"112 选 7 的困难精排"。HyDE 和 Graph 解决的是同一问题的不同侧面（语义对齐 vs 跨文档关联），叠加使用存在架构不兼容。正确做法是**路由策略**：单跳用 HyDE，多跳用 Graph+CoT。

### Q12: vector_filter_k 是什么？为什么有效？（v7 核心创新）

**A**: vector_filter_k 是我们在 v7 中引入的「拓扑 × 语义双重门控」机制。

**问题**：BFS 扩展的邻居是图上的拓扑邻居（共享 bridge term），但不一定和当前 query 语义相关。

**解决方案**：先做一次宽泛的向量检索（k=50），得到语义相关 chunk 白名单。BFS 扩展时，只保留**同时在白名单中**的邻居——相当于给图扩展加了一个语义门控。

**效果**：chain_score 从 0.697(v3_cot) → 0.729(v7_vf50)，+4.6%。vf=50 是最佳平衡点。

**面试话术**："图的拓扑连接保证跨文档发现（exploration），向量白名单保证 query 相关性（exploitation），交集就是'既跨书又语义相关'的高质量候选。"

### Q13: 你是怎么发现和修复评估体系问题的？

**A**: 发现了两个系统性偏差：

1. **Self-eval bias**：Kimi 做 Judge 比 GPT-4o 宽松 10~33%。修复：引入 GPT-4o 作为第三方 Judge 重评所有配置。
2. **英文 Eval Embedding**：RAGAS 默认用英文 all-MiniLM-L6-v2 衡量中文 relevancy → 系统性低估（差 12.5pp）。修复：指定中文 bge-small-zh-v1.5。

**面试话术**："'如何评估'有时比'如何优化'更重要。修复后分数降了 10~33%，这恰恰证明修复前的高分是虚胖。"

### Q14: CoT Prompt 为什么对多跳推理效果这么大？

**A**: 因为多跳推理的瓶颈不在于 LLM 的能力，而在于 LLM 默认的回答策略：

- **没有 CoT 时**：LLM 倾向于给出一个综合性的直接回答，往往只融合了最相关的 1-2 本书的内容，跳过了中间推理步骤
- **有 CoT 时**：强制 LLM 先列出每本书的相关要点，再逐步连接它们的逻辑关联，最后得出结论
- 这类似于人类做多跳推理——如果直接问"A和B什么关系"，可能答不好；但如果提示"先分析A的观点、再分析B的观点、最后比较"，答案质量会大幅提升

实测数据显示：CoT 让之前因"检索到了但没整合"而失分的题目大幅提升。这些题的检索结果和非 CoT 版本完全一样——只是 LLM 被引导后更好地利用了已有的 context。

### Q15: 你的生产部署推荐是什么？

**A**: **路由策略**——根据问题类型选择最优管道：

| 问题类型 | 推荐方案 | 原因 |
|---------|---------|------|
| 单跳/单书 | HyDE + Rerank (AVG 0.812) | 假设古文填补词汇鸿沟，精排简单高效 |
| 跨书多跳 | Graph RAG v7 vf50 (chain 0.729) | IDF 图 + 语义门控有效桥接不同古籍 |

路由可以用简单的关键词分类器（问题中是否提到多本书或比较类词汇），或让 LLM 做一次短分类。

---

*文档更新日期：2026-04-19（GPT-4o 全面重评 + v6-v8 演进 + 评估修复 + Prompt Tuning 天花板 + STAR 挑战故事集）*

---

## 15. 核心代码速查

以下代码块来自真实项目文件，面试中可用于解释关键逻辑。

### 15.1 HyDE + Rerank 检索（`fortune_langchain_utils.py`）

```python
HYDE_PROMPT = (
    "你是中国传统命理学专家。请根据以下问题，"
    "仿照古典命理文献（文言文/半文言文）的风格，"
    "写一段80-150字的原文片段，直接包含问题答案所涉及的术语和论述。"
    "只输出片段本身，不要标题、序号或解释。\n\n"
    "问题：{question}"
)

def _build_hyde_rerank_retriever(llm, hyde_k=15, top_n=7):
    vectorstore = get_vectorstore()

    def _retrieve(query) -> list[Document]:
        q = query.get("input", "") if isinstance(query, dict) else str(query)

        # Step 1: HyDE — LLM 生成仿古文
        hyp_text = llm.invoke(HYDE_PROMPT.format(question=q)).content.strip()

        # Step 2: 用假设文档 embedding 做宽召回
        candidates = vectorstore.similarity_search(hyp_text, k=hyde_k)

        # Step 3: 用原始问题 + CrossEncoder 精排
        encoder = _get_bge_encoder()
        pairs  = [[q, doc.page_content] for doc in candidates]
        scores = encoder.score(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_n]]

    return RunnableLambda(_retrieve)
```

### 15.2 Graph RAG CrossEncoder 精排（`graph_retriever.py`）

```python
# Step 4: 对全量候选 (seeds + graph neighbors) 做 CrossEncoder 精排
tok, mdl = self._get_reranker()
with torch.no_grad():
    inputs = tok(
        [query] * len(candidates),
        [d.page_content for d in candidates],
        return_tensors="pt",
        truncation=True,
        max_length=256,       # query+doc 合计最多 256 tokens
        padding=True,
    )
    scores = mdl(**inputs).logits.squeeze(-1).tolist()

ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
selected = _diverse_top_n(ranked, self.top_n)
```

### 15.3 分块函数（`build_index_bge.py`）

```python
# 三命通会：按 ○ 标记切分
def split_by_circle_marker(text: str) -> list[str]:
    return [c.strip() for c in re.split(r"(?=○)", text) if c.strip()]

# 子平真诠：按汉字章节号切分
def split_by_chapter_number(text: str) -> list[str]:
    return [c.strip() for c in
            re.split(r"(?=\n[一二三四五六七八九十百千]+[、．。])", text)
            if c.strip()]

# 滴天髓：按空行切分 + 四字诀合并原注/任氏曰
def split_by_blank_line(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    merged = []
    i = 0
    while i < len(paragraphs):
        para = paragraphs[i]
        is_verse = (len(para) <= 100
                    and not para.startswith("原注")
                    and not para.startswith("任氏曰"))
        if is_verse and i + 1 < len(paragraphs):
            combined = para
            j = i + 1
            while j < len(paragraphs):
                # ...合并后续段落到下一个四字诀或超出 1000 字...
                j += 1
            merged.append(combined)
            i = j
        else:
            merged.append(para)
            i += 1
    return merged
```

### 15.4 知识图谱构建（`build_knowledge_graph.py`）

```python
BRIDGE_TERMS = ["官星","财星","印绶","伤官","格局","用神","日主","五行", ...]  # 36 个

# 倒排索引 + IDF 计算
term_to_chunks: dict[str, list[str]] = defaultdict(list)
for cid, info in chunk_index.items():
    for term in set(info["bridge_terms"]):
        term_to_chunks[term].append(cid)

N = len(chunk_index)  # 668
term_idf = {term: math.log(N / len(cids)) for term, cids in term_to_chunks.items()}

# 枚举跨书边（IDF 加权）
edge_weight: dict[tuple, float] = defaultdict(float)
edge_terms:  dict[tuple, list] = defaultdict(list)

for term, cids in term_to_chunks.items():
    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            a, b = cids[i], cids[j]
            if chunk_index[a]["book"] == chunk_index[b]["book"]:
                continue
            key = (min(a, b), max(a, b))
            edge_weight[key] += term_idf[term]   # IDF 加权
            edge_terms[key].append(term)

# 写入边（min_weight + 度限制）
for (a, b), w in edge_weight.items():
    if w >= min_weight:  # min_weight=3 (IDF 阈值)
        G.add_edge(a, b, weight=round(w, 4), shared_terms=edge_terms[(a, b)])

# Union 度限制剪枝 (max_degree=15)
if max_degree > 0:
    for node in list(G.nodes):
        edges = sorted(G.edges(node, data=True), key=lambda e: e[2]["weight"], reverse=True)
        keep_set = {(min(u,v), max(u,v)) for u, v, _ in edges[:max_degree]}  # 每端点 top-K
    # Union: 边只要被任一端点保留就留下
    ...
```

---

## 16. 项目文件结构

```
Chinese-Fortune-Telling/
├── api/                              # FastAPI 后端
│   ├── main.py                       # 通用 RAG API（非命理专用）
│   ├── fortune_main.py               # 命理 RAG API（生产入口）
│   ├── fortune_langchain_utils.py    # HyDE + Rerank 管道（生产检索）
│   ├── fortune_prompts.py            # 所有 Prompt 定义（角色/评估/分析）
│   ├── graph_retriever.py            # Graph RAG 检索器（仅 Benchmark）
│   ├── chroma_utils.py               # ChromaDB 连接与向量库工具
│   ├── pydantic_models.py            # 请求/响应数据模型
│   └── requirements.txt
├── app/                              # Streamlit 前端
│   ├── fortune_app.py                # 命理聊天主界面
│   ├── fortune_chat_interface.py     # 聊天 UI 组件
│   ├── fortune_sidebar.py            # 侧边栏配置
│   ├── fortune_api_utils.py          # 前端 → API 调用封装
│   └── requirements.txt
├── scripts/                          # 离线脚本
│   ├── build_index_bge.py            # 分块 + BGE 编码 → ChromaDB
│   ├── build_knowledge_graph.py      # 桥接术语 → NetworkX 图
│   ├── rag_bench.py                  # RAGAS 单跳 Benchmark
│   ├── bench_multihop.py             # 多跳推理 Benchmark
│   ├── generate_qa_dataset.py        # 单跳 QA 数据集生成
│   ├── generate_qa_multihop.py       # 多跳 QA 数据集生成
│   └── show_results.py              # Benchmark 结果可视化
├── configs/rag/                      # 检索策略 YAML 配置
│   ├── baseline.yaml
│   ├── hybrid.yaml
│   ├── hybrid_rerank_bge.yaml        # 含 ms-marco 失败记录
│   └── ...
├── benchmarks/                       # 评估数据集 + 结果
│   ├── qa_dataset.json               # 50 题单跳
│   ├── qa_multihop.json              # 36 题多跳
│   └── results/                      # JSON 评估详细结果
├── chroma_db_bge/                    # ChromaDB 持久化目录（668 vectors）
├── data/
│   ├── chunk_index.json              # chunk_id → {book, text, bridge_terms}
│   └── graph_stats.json              # 图统计信息
├── fortune_books/                    # 原始古籍文本 + metadata.json
├── terraform/                        # Azure IaC 部署
├── docker-compose.yml                # 容器编排（API + App）
└── deploy.sh                         # 一键部署脚本
```

---

## 17. STAR 挑战故事集（面试行为面必备）

> **使用方法**：面试官问"遇到什么挑战/How did you solve X"时，从以下 6 个故事中选最相关的讲。每个故事按 STAR 法则（Situation → Task → Action → Result）结构化，控制在 2 分钟内讲完。

---

### STAR 1: 跨语言 Reranker 踩坑 — 英文模型给古汉语随机打分

**Situation**

RAG 系统检索的是古典命理文献（文言文/半文言文），向量检索 + BM25 混合召回后需要 CrossEncoder 精排来提升最终返回文档的质量。

**Task**

引入 Cross-Encoder reranker，使精排后的 top-k 文档比粗排更精准。

**Action**

1. 初始选择了 `cross-encoder/ms-marco-MiniLM-L-6-v2`——Stack Overflow 和 SentenceTransformer 文档推荐的默认 reranker，在英文 MSMARCO 数据集上训练
2. 上线后发现 **Rerank 后效果反而变差**：原先 EnsembleRetriever（BM25 + 向量）通过 RRF 融合已排出合理顺序，加入 english reranker 后排序被打乱
3. **Root cause 诊断**：ms-marco 的 tokenizer 和训练语料完全是英文，对古汉语 token 几乎无语义理解，输出的 logit 分数接近随机——等于把 RRF 已经排好的序列随机 shuffle 了一遍
4. 切换到 `BAAI/bge-reranker-base`（XLMRoberta 架构，中文数据训练的 278M Cross-Encoder），重新评估

**Result**

- 精排从「负面效果」变为「正面效果」，最终 HyDE + BGE Rerank 方案达到单跳 AVG = **0.812**（全系统最优）
- faithfulness 从 0.762（无 rerank baseline）→ **0.917**
- **教训写入了配置文件注释**（`configs/rag/hybrid_rerank_bge.yaml`），作为团队 knowledge base

> **面试追问准备**：
> - "怎么发现是 reranker 的问题而不是其他环节？" → 对比 with/without reranker 的结果，发现去掉 reranker 反而更好，定位到 reranker 是唯一变量
> - "为什么不一开始就用中文 reranker？" → 当时对 Cross-Encoder 生态不熟，ms-marco 是社区最常推荐的开箱即用方案

---

### STAR 2: 知识图谱噪声爆炸 — 从 43K 条边到 5.3K 条边

**Situation**

为支持跨书多跳推理，构建了一张以 chunk 为节点、共享命理术语为边的知识图谱。v1 版本有 668 个节点和 **43,000+ 条边**（平均度 47），BFS 一步扩展就产生海量噪声候选，淹没了 reranker 的精排能力。chain_score 只有 **0.435**。

**Task**

在保留有价值的跨书关联前提下，大幅降低图的噪声密度，使 BFS 扩展的候选质量显著提升。

**Action**

设计了三层递进的过滤机制：

1. **IDF 加权边权重**：高频术语（如「五行」出现在 158/668 个 chunk 中，IDF≈1.4）贡献极低权重，稀有术语（如「调候」df=12，IDF≈3.9）贡献高权重。两个 chunk 共享了高频术语不代表真的相关，共享稀有术语才说明有实质性关联
2. **min_weight=3 阈值**：IDF 累计权重 < 3 的边直接丢弃。例如两个 chunk 只共享「五行」+「官星」（IDF=1.4+1.4=2.8 < 3）→ 不建边
3. **Union max_degree=15 剪枝**：每个节点只保留权重最高的 top-15 条边。Union 策略（边只要被任一端点保留就留下）比 Intersection 更温和，避免过度剪枝

**Result**

| 指标 | v1 | v3（IDF+剪枝） | 变化 |
|------|-----|---------------|------|
| 边数 | 43,000+ | **5,255** | **-87.8%** |
| 平均度 | 47.0 | **15.7** | -66.6% |
| chain_score | 0.435 | **0.546** | **+25.5%** |
| cross_book_hit | 69.4% | **83.3%** | +13.9pp |

- 核心洞察：**图的价值不在于边多，而在于边的信息量高**。IDF 加权让图从「广而浅」变为「窄而深」

> **面试追问准备**：
> - "为什么用 IDF 不用 TF-IDF？" → chunk 内部已去重（`set(bridge_terms)`），每个术语 TF=1，TF-IDF 退化为 IDF
> - "max_degree=15 怎么选的？" → 试了 10/15/20，15 是 cross_book_hit 和 chain_score 的帕累托最优点
> - "为什么只建跨书边？" → 同书 chunk 向量检索已覆盖，graph 的价值是补跨书关联；同书边会挤占 max_degree 预算

---

### STAR 3: 评估体系的两个系统性偏差 — 「既当运动员又当裁判」

**Situation**

项目初期，RAG 生成用 Kimi LLM，RAGAS 评估的 Judge LLM **也用 Kimi**，评估 Embedding 使用 RAGAS 默认的 **英文 all-MiniLM-L6-v2**。随着迭代到 v3-v4，分数持续上涨，但直觉上感觉评估可能有水分。

**Task**

验证评估体系是否存在系统性偏差，如果存在则修复并用修正后的分数重新排名。

**Action**

1. **假设 1：Self-eval bias** — Kimi 评自己的回答可能系统性偏宽松
   - 引入 GPT-4o 作为第三方 Judge，对**全部 19 个 multihop 配置 + 9 个 normal 配置**做 rescore
   - 结果：**faithfulness 平均被扣 10~15pp，multihop chain_score 被扣 5~13pp**
   - 确认 Kimi self-eval 系统性偏乐观

2. **假设 2：英文 Eval Embedding 对中文 relevancy 失真** — 在对比 rescore 结果时偶然发现
   - RAGAS 的 answer_relevancy 内部用 embedding cosine 衡量答案和问题的语义距离
   - 英文 all-MiniLM-L6-v2 处理中文/古汉语 → 系统性低估 relevancy
   - 换成中文 bge-small-zh-v1.5 后 relevancy 从 0.524 → **0.656**（+12.5pp）

3. 用修正后的评估体系（GPT-4o Judge + 中文 Eval Embedding）重评所有配置，产出最终排行榜

**Result**

- 修复后**配置排名没有大幅变化**（v7 仍是多跳最优、HyDE+Rerank 仍是单跳最优），说明相对排名是稳健的
- 但**绝对分数下降 10~33%**，证明修复前的高分确实是虚胖
- **关键教训**：「如何评估」有时比「如何优化」更重要。如果评估本身不可信，所有基于评估的优化决策都可能是错的

> **面试追问准备**：
> - "怎么发现英文 embedding 的问题？" → 是在 GPT-4o rescore 过程中对比 relevancy 分数时偶然发现的，不是主动排查
> - "为什么不直接用 GPT-4o 做生成？" → 成本考虑（Kimi 免费额度高），且 RAG 的核心是检索而非生成——换 LLM 对检索指标（recall/precision）没影响

---

### STAR 4: HyDE + Graph 架构不兼容 — 1+1 < 1 的负面实验

**Situation**

HyDE+Rerank 是单跳最优（AVG 0.812），Graph RAG 是多跳最优（chain 0.697）。自然的想法是：**把两者结合**——用 HyDE 提升 seed 质量，再走 Graph BFS 展开跨书邻居，岂不是两全其美？

**Task**

验证 HyDE + Graph 组合的效果，如果有效则作为统一方案，如果无效则找出 root cause。

**Action**

1. **构建 v5 方案**：`question → HyDE 生成 → similarity_search(hyde_text, k=10) → BFS 邻居展开 → CrossEncoder 精排(original_query) → top-7`
2. **实验结果令人意外**：
   - 多跳 chain_score：v5 = 0.671 < v3_cot = 0.697（**−3.7pp**）
   - 单跳 AVG：v5 = 0.779 < 纯 HyDE = 0.812（**−3.3pp**）
   - 两个场景都变差了
3. **三层深度诊断**：
   - **文档重叠率分析**：v5 最终结果和 v3（无 HyDE 的 Graph）的重叠率高达 **64.9%**，但和纯 HyDE 的重叠率仅 **0.6%** → 说明 Graph 展开完全覆盖了 HyDE 的改善
   - **Seed/Neighbor 量化分析**（20 题逐题追踪）：seed 仅占候选池 **8.9%**（10/112），但被 reranker 放大到 top-7 中的 **54.3%**（6 倍放大）。seed 均分 −0.206 vs neighbor 均分 −2.477 → reranker 在努力过滤噪音，但 45.7% 的 neighbor 仍然渗入
   - **提出框架解释**：纯 HyDE 是「15 选 5 的简单精排问题」；Graph 把它变成了「112 选 7 的困难精排问题」。278M 的 bge-reranker-base 区分力不足以处理后者

**Result**

- 证明 HyDE 和 Graph RAG 存在**架构层面的不兼容**：它们解决同一问题（query-document gap）的不同侧面，叠加使用导致信息损失
- 提出**路由策略**作为正确架构：单跳用 HyDE+Rerank，多跳用 Graph+CoT，不做简单叠加
- **这是项目中最有面试价值的负面实验**——证明工程师不是盲目叠加技术，而是理解每种技术的适用边界

> **面试追问准备**：
> - "有没有可能是实现问题而不是架构问题？" → 排除了——v5 和 v3 共享完全相同的 BFS + reranker 代码，唯一变量是 seed 来源
> - "如果 reranker 更强呢？" → 试了 v2-m3（568M）和 reranker-large（560M），效果反而更差。问题不在 reranker 能力，而在候选池质量分布

---

### STAR 5: Prompt 工程 ROI 远超模型升级 — 挑战「大力出奇迹」直觉

**Situation**

Graph RAG v3（IDF 加权图）的 chain_score 达到 0.546，图谱质量已大幅优化。接下来需要决定资源投入方向：是继续优化检索，还是升级模型，还是调 prompt？

**Task**

找到性价比最高的提升路径，在有限的 API 预算和时间内最大化 chain_score。

**Action**

同时做了两组对比实验：

1. **Prompt 方向（CoT）**：设计 Chain-of-Thought prompt，强制 LLM 按「各书要点 → 逐步推理 → 结论」的三段式作答。不改任何检索逻辑，只改 system prompt 中的指令
2. **模型方向**：
   - 升级 reranker 从 bge-reranker-base（278M）→ bge-reranker-v2-m3（568M，多语言 SOTA）
   - 升级 embedding 从 bge-small-zh（384d）→ bge-base-zh（768d）
3. **天花板测试**：在最优检索配置上测试 3 种 prompt 风格（default / concise / balanced），验证 prompt tuning 边际收益

**Result**

| 优化方向 | chain_score 变化 | 成本 |
|---------|:---:|------|
| CoT Prompt | **+27.7%**（0.546→0.697） | 0（改 prompt 文本） |
| 升级 reranker v2-m3 | **−13.8%**（0.697→0.637） | 模型下载 + 推理翻倍 |
| 升级 embedding bge-base | +4.0%（0.697→0.725） | 重建整个向量索引 |
| Prompt 天花板测试 | ≤ 2pp 差异 | — |

- **核心教训**：小规模专业领域中，**prompt 工程的 ROI 远超模型升级**。检索给了正确的 context，LLM 只需要被正确引导就能整合信息
- **大 reranker 反而更差的原因**：bge-reranker-v2-m3 是多语言通用模型（支持 100+ 语种），在古典中文上的排序不如中文专精的 bge-reranker-base。**没有银弹**
- **Prompt 天花板**：3 种 prompt 风格差异 ≤ 2pp，说明 prompt tuning 已触顶，下一步收益必须来自检索质量

> **面试追问准备**：
> - "为什么 CoT 对多跳特别有效？" → 多跳瓶颈不在检索而在 LLM 的整合策略。没有 CoT 时 LLM 倾向只综合 1-2 本书；CoT 强制它先列出每本书要点再找关联
> - "prompt tuning 触顶后怎么办？" → 瓶颈转移到检索端，所以设计了 vector_filter_k（STAR 6）

---

### STAR 6: 拓扑×语义双重门控 — vector_filter_k 的诞生（v7 核心创新）

**Situation**

CoT prompt 之后 chain_score = 0.697。分析失败案例发现：BFS 按静态 IDF 权重排序邻居，**不感知当前 query 的语义**——可能走了一条 IDF 很高但与当前问题无关的边，浪费了 max_neighbors 的有限预算。

**Task**

设计一种机制，让 BFS 展开的邻居**既满足图的拓扑关联（跨书桥接），又满足与当前 query 的语义相关性**。

**Action**

1. **核心思路**：先对 query 做一次宽泛的向量检索（k=50），得到语义相关 chunk 的「白名单」。BFS 展开时，只保留**同时出现在白名单中**的邻居
2. **实现**：在 `_bfs_neighbors()` 中增加一行检查 `if nbr not in vector_whitelist: continue`
3. **消融实验**：测试 vf_k=0（不过滤）/ 50 / 100
   - vf=50：**0.729**（最优）
   - vf=100：0.720（白名单太宽，过滤效果减弱）
   - vf=0：0.697（纯 BFS 无语义过滤）

**Result**

| 指标 | v3_cot（无 vf） | v7_vf50 | 变化 |
|------|:---:|:---:|:---:|
| chain_score | 0.697 | **0.729** | **+4.6%** |
| hop_ok | 69.4% | **72.2%** | +2.8pp |
| full_ok | 44.4% | **47.2%** | +2.8pp |

- v7_vf50 成为**多跳 benchmark 的最终冠军配置**
- **设计优雅之处**：不引入任何新超参数（50 是图的 top-n 邻居自然覆盖范围）、不改 BFS 算法主体、不增加额外模型——只加了一行向量检索作为 gate

> **面试话术**："图的拓扑连接保证跨文档发现（exploration），向量白名单保证 query 相关性（exploitation），交集就是'既跨书又语义相关'的高质量候选。"
>
> **面试追问准备**：
> - "vector_filter_k=50 这个值怎么选？" → 50 约等于全库 668 chunks 的 7.5%，是 top-n 语义相关的自然边界；100 太宽等于不过滤，25 太严会误伤有效邻居
> - "这和直接用向量检索有什么区别？" → 向量检索只返回语义相似 chunk，不保证跨书覆盖；graph BFS 保证跨书但不保证 query 相关。vector_filter_k 取交集，两个都保证
