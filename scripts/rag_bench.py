"""
rag_bench.py
~~~~~~~~~~~~
对中文命理 RAG 系统进行多配置基准测试，使用 RAGAS 指标评估。

支持的检索配置（通过 --configs 指定 YAML 文件）:
  - baseline  : 纯向量检索 k=3
  - top_k     : 纯向量检索 k=5 / k=10
  - hybrid    : BM25 + 向量混合检索
  - rerank    : 向量检索 + 交叉编码器重排
  - rag_fusion: 多查询融合检索

用法:
    # 跑单个配置
    python scripts/rag_bench.py --config configs/rag/baseline.yaml

    # 跑所有配置并生成对比报告
    python scripts/rag_bench.py \
        --configs configs/rag/baseline.yaml configs/rag/hybrid.yaml configs/rag/rerank.yaml \
        --dataset benchmarks/qa_dataset.json \
        --output-dir benchmarks/results

评估指标:
    - faithfulness       (回答与检索上下文的事实一致性)
    - answer_relevancy   (回答与问题的相关度)
    - context_recall     (检索到的上下文覆盖黄金答案的程度)
    - context_precision  (检索上下文中有用内容的比例)
    - latency_p50/p95    (检索+生成延迟，秒)
    - tokens_used        (生成阶段 token 消耗，若模型支持)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# 把 api/ 加入路径
sys.path.insert(0, str(Path(__file__).parent.parent / "api"))

from dotenv import load_dotenv

load_dotenv(override=True)


# ---------------------------------------------------------------------------
# LLM 工厂（支持 kimi / gemini / groq / deepseek）
# ---------------------------------------------------------------------------


def build_llm(cfg: dict):
    """
    根据 cfg['generation']['provider'] 构建 LLM。

    YAML 示例:
        generation:
          provider: kimi            # kimi | gemini | groq | deepseek
          model: moonshot-v1-32k
          temperature: 0.7
    """
    from langchain_openai import ChatOpenAI

    gen_cfg = cfg.get("generation", {})
    provider = gen_cfg.get("provider", "kimi")
    model = gen_cfg.get("model", "moonshot-v1-32k")
    temperature = float(gen_cfg.get("temperature", 0.7))

    if provider == "kimi":
        api_key = os.environ.get("KIMI_API_KEY")
        if not api_key:
            raise EnvironmentError("KIMI_API_KEY not set in .env")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url="https://api.moonshot.cn/v1",
            api_key=api_key,
        )

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, max_retries=3)

    elif provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not set in .env")
        return ChatGroq(model=model, temperature=temperature, api_key=api_key)

    elif provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise EnvironmentError("DEEPSEEK_API_KEY not set in .env")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url="https://api.deepseek.com",
            api_key=api_key,
        )

    else:
        raise ValueError(f"Unknown provider: '{provider}'. Choose: kimi | gemini | groq | deepseek")


# ---------------------------------------------------------------------------
# 检索器工厂
# ---------------------------------------------------------------------------


def build_retriever(cfg: dict, vectorstore):
    """
    根据配置构建 LangChain retriever。

    cfg 示例:
        retrieval:
          type: hybrid        # vector | bm25 | hybrid | rerank | rag_fusion
          k: 5
          rerank_model: cross-encoder/ms-marco-MiniLM-L-6-v2
          bm25_weight: 0.5    # hybrid 时 BM25 权重
    """
    retrieval_cfg = cfg.get("retrieval", {})
    r_type = retrieval_cfg.get("type", "vector")
    k = int(retrieval_cfg.get("k", 3))

    # ── 纯向量 ──────────────────────────────────────────────────────────────
    if r_type == "vector":
        return vectorstore.as_retriever(search_kwargs={"k": k})

    # ── 纯 BM25 ─────────────────────────────────────────────────────────────
    elif r_type == "bm25":
        from langchain_community.retrievers import BM25Retriever

        docs = vectorstore.get()["documents"]
        from langchain_core.documents import Document

        raw_docs = [Document(page_content=t) for t in docs if t]
        return BM25Retriever.from_documents(raw_docs, k=k)

    # ── 混合（BM25 + 向量 EnsembleRetriever）─────────────────────────────────
    elif r_type == "hybrid":
        from langchain_classic.retrievers import EnsembleRetriever
        from langchain_community.retrievers import BM25Retriever
        from langchain_core.documents import Document

        bm25_w = float(retrieval_cfg.get("bm25_weight", 0.5))
        vec_w = 1.0 - bm25_w

        docs_result = vectorstore.get()
        raw_docs = [Document(page_content=t, metadata=m) for t, m in zip(docs_result["documents"], docs_result["metadatas"]) if t]
        bm25 = BM25Retriever.from_documents(raw_docs, k=k)
        vec = vectorstore.as_retriever(search_kwargs={"k": k})
        return EnsembleRetriever(retrievers=[bm25, vec], weights=[bm25_w, vec_w])

    # ── 向量 + 交叉编码器重排 ─────────────────────────────────────────────────
    elif r_type == "rerank":
        from langchain_classic.retrievers import ContextualCompressionRetriever
        from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder

        rerank_model = retrieval_cfg.get("rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        top_n = int(retrieval_cfg.get("top_n", 3))
        fetch_k = int(retrieval_cfg.get("fetch_k", k * 3))

        base_retriever = vectorstore.as_retriever(search_kwargs={"k": fetch_k})
        encoder = HuggingFaceCrossEncoder(model_name=rerank_model)
        compressor = CrossEncoderReranker(model=encoder, top_n=top_n)
        return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)

    # ── 混合检索 + 交叉编码器精排（hybrid_rerank）────────────────────────────────
    # 步骤 1：BM25 + 向量 EnsembleRetriever 宽召回（每路各取 k 个，合并去重后约 k~2k 个候选）
    # 步骤 2：CrossEncoderReranker 对每个候选单独计算问题-文档相关分，保留 top_n
    # 优势：兼具 hybrid 的高召回（古汉语字面/语义双路）和 rerank 的高精度（精排）
    elif r_type == "hybrid_rerank":
        from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
        from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
        from langchain_community.retrievers import BM25Retriever
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        from langchain_core.documents import Document

        bm25_w = float(retrieval_cfg.get("bm25_weight", 0.4))
        vec_w = 1.0 - bm25_w
        top_n = int(retrieval_cfg.get("top_n", 3))
        rerank_model = retrieval_cfg.get("rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")

        # Step 1: build hybrid base retriever
        docs_result = vectorstore.get()
        raw_docs = [Document(page_content=t, metadata=m) for t, m in zip(docs_result["documents"], docs_result["metadatas"]) if t]
        bm25 = BM25Retriever.from_documents(raw_docs, k=k)
        vec = vectorstore.as_retriever(search_kwargs={"k": k})
        hybrid_retriever = EnsembleRetriever(retrievers=[bm25, vec], weights=[bm25_w, vec_w])

        # Step 2: wrap with cross-encoder reranker
        encoder = HuggingFaceCrossEncoder(model_name=rerank_model)
        compressor = CrossEncoderReranker(model=encoder, top_n=top_n)
        return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=hybrid_retriever)

    # ── 命题检索（prop_vector / prop_hybrid）────────────────────────────────
    # 检索时：搜索命题索引（粒度细 → 精准）
    # 生成时：返回命题对应的父块原文（粒度粗 → 上下文完整）
    elif r_type in ("prop_vector", "prop_hybrid"):
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
        from langchain_core.runnables import RunnableLambda
        from langchain_huggingface import HuggingFaceEmbeddings

        emb_model     = cfg.get("embedding_model", "all-MiniLM-L6-v2")
        prop_dir      = cfg.get("prop_chroma_dir", "./chroma_db_prop")
        top_k_props   = int(retrieval_cfg.get("top_k_props", 15))
        top_k_parents = int(retrieval_cfg.get("top_k_parents", k))
        bm25_w        = float(retrieval_cfg.get("bm25_weight", 0.4))

        # 加载命题向量库（与主索引共享相同嵌入模型）
        prop_emb = HuggingFaceEmbeddings(
            model_name=emb_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        prop_vs = Chroma(persist_directory=prop_dir, embedding_function=prop_emb)

        def _prop_retrieve(query) -> list[Document]:
            """搜命题 → 去重 → 返回父块 Documents。
            langchain_classic 对非 BaseRetriever 的 Runnable 传入整个 dict，
            兼容处理 str | dict 两种输入。
            """
            q = query.get("input", "") if isinstance(query, dict) else query
            p_docs = prop_vs.similarity_search(q, k=top_k_props)
            seen: set = set()
            parents: list[Document] = []
            for doc in p_docs:
                parent_text = doc.metadata.get("parent_chunk", doc.page_content)
                key = hash(parent_text[:80])
                if key not in seen:
                    seen.add(key)
                    meta = {
                        mk: mv
                        for mk, mv in doc.metadata.items()
                        if mk != "parent_chunk"
                    }
                    parents.append(Document(page_content=parent_text, metadata=meta))
                if len(parents) >= top_k_parents:
                    break
            return parents

        if r_type == "prop_vector":
            return RunnableLambda(_prop_retrieve)

        # prop_hybrid: BM25（原始块字面匹配）+ 命题向量（语义匹配父块）→ 手动 RRF 合并
        from langchain_community.retrievers import BM25Retriever

        docs_result = vectorstore.get()
        raw_docs = [
            Document(page_content=t, metadata=m)
            for t, m in zip(docs_result["documents"], docs_result["metadatas"])
            if t
        ]
        bm25 = BM25Retriever.from_documents(raw_docs, k=k)

        def _prop_hybrid_retrieve(query) -> list[Document]:
            q = query.get("input", "") if isinstance(query, dict) else query
            bm25_docs  = bm25.invoke(q)
            prop_docs  = _prop_retrieve(q)
            # RRF 合并：每路各贡献 1/(rank+60)
            scores: dict[int, float] = {}
            id_to_doc: dict[int, Document] = {}
            for rank, doc in enumerate(bm25_docs):
                key = hash(doc.page_content[:80])
                scores[key]    = scores.get(key, 0.0) + 1.0 / (rank + 60)
                id_to_doc[key] = doc
            for rank, doc in enumerate(prop_docs):
                key = hash(doc.page_content[:80])
                scores[key]    = scores.get(key, 0.0) + 1.0 / (rank + 60)
                id_to_doc[key] = doc
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return [id_to_doc[key] for key, _ in ranked[:top_k_parents]]

        return RunnableLambda(_prop_hybrid_retrieve)

    # ── HyDE（Hypothetical Document Embedding）────────────────────────────────
    # 核心思想：现代汉语问法 ≠ 古典汉语文体 → 直接用问题向量检索效果差。
    # 解法：让 LLM 先生成"古籍风格的假设答案片段"，用该片段的向量检索，
    #       命中概率远高于用问题本身。最终仍返回检索到的原始文档块给 LLM。
    #
    # hyde        : 假设文档向量检索（单路，纯向量）
    # hyde_hybrid : 假设文档向量 + BM25（原始问题字面匹配）→ RRF 融合
    elif r_type in ("hyde", "hyde_hybrid"):
        from langchain_core.documents import Document
        from langchain_core.runnables import RunnableLambda
        from langchain_openai import ChatOpenAI

        hyde_k       = k
        bm25_w       = float(retrieval_cfg.get("bm25_weight", 0.4))
        num_hyp      = int(retrieval_cfg.get("num_hypothetical", 1))

        # HyDE 生成用低温度、快速模型（moonshot-v1-8k）
        gen_cfg      = cfg.get("generation", {})
        kimi_key     = os.environ.get("KIMI_API_KEY")
        hyde_llm     = ChatOpenAI(
            model="moonshot-v1-8k",
            temperature=0.3,
            max_tokens=400,
            base_url="https://api.moonshot.cn/v1",
            api_key=kimi_key,
        )

        # Prompt：引导 LLM 生成"古籍原文风格"的假设文档片段
        HYDE_PROMPT = (
            "你是中国传统命理学专家。请根据以下问题，"
            "仿照古典命理文献（文言文/半文言文）的风格，"
            "写一段 80-150 字的原文片段，直接包含问题答案所涉及的术语和论述。"
            "只输出片段本身，不要标题、序号或解释。\n\n"
            "问题：{question}"
        )

        def _hyde_retrieve(query) -> list[Document]:
            q = query.get("input", "") if isinstance(query, dict) else query
            # 生成 num_hyp 条假设文档（通常 1 条足够）
            hyp_texts: list[str] = []
            for _ in range(num_hyp):
                try:
                    hyp = hyde_llm.invoke(HYDE_PROMPT.format(question=q)).content.strip()
                    hyp_texts.append(hyp)
                except Exception:
                    hyp_texts.append(q)   # 生成失败时退回原始问题
            # 取所有假设文档的向量检索结果并去重
            seen: set = set()
            merged: list[Document] = []
            for hyp_text in hyp_texts:
                for doc in vectorstore.similarity_search(hyp_text, k=hyde_k):
                    key = hash(doc.page_content[:80])
                    if key not in seen:
                        seen.add(key)
                        merged.append(doc)
            return merged[:hyde_k]

        if r_type == "hyde":
            return RunnableLambda(_hyde_retrieve)

        # hyde_hybrid: BM25（原始问题字面匹配）+ HyDE向量（语义补全）→ RRF
        from langchain_community.retrievers import BM25Retriever

        docs_result = vectorstore.get()
        raw_docs = [
            Document(page_content=t, metadata=m)
            for t, m in zip(docs_result["documents"], docs_result["metadatas"])
            if t
        ]
        bm25_hyde = BM25Retriever.from_documents(raw_docs, k=hyde_k)

        def _hyde_hybrid_retrieve(query) -> list[Document]:
            q = query.get("input", "") if isinstance(query, dict) else query
            bm25_docs = bm25_hyde.invoke(q)
            hyde_docs = _hyde_retrieve(q)
            scores: dict[int, float] = {}
            id_to_doc: dict[int, Document] = {}
            for rank, doc in enumerate(bm25_docs):
                key = hash(doc.page_content[:80])
                scores[key]    = scores.get(key, 0.0) + 1.0 / (rank + 60)
                id_to_doc[key] = doc
            for rank, doc in enumerate(hyde_docs):
                key = hash(doc.page_content[:80])
                scores[key]    = scores.get(key, 0.0) + 1.0 / (rank + 60)
                id_to_doc[key] = doc
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return [id_to_doc[key] for key, _ in ranked[:hyde_k]]

        return RunnableLambda(_hyde_hybrid_retrieve)

    # ── HyDE + BGE 交叉编码器精排（hyde_rerank）──────────────────────────────────
    # 步骤 1：HyDE 生成假设文档片段，用其向量宽召回 k 个候选（通常 k=15）
    # 步骤 2：BGE 中文交叉编码器对"原始问题 vs 每个候选文档"逐一打分，保留 top_n
    # 优势：HyDE 解决词汇鸿沟（现代汉语 → 古典文献），BGE 精排去除主题偏移的假设文档噪声
    elif r_type == "hyde_rerank":
        from langchain_core.documents import Document
        from langchain_core.runnables import RunnableLambda
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        from langchain_openai import ChatOpenAI

        top_n        = int(retrieval_cfg.get("top_n", 5))
        hyde_k       = k                                       # 宽召回数量（默认 15）
        num_hyp      = int(retrieval_cfg.get("num_hypothetical", 1))
        rerank_model = retrieval_cfg.get("rerank_model", "BAAI/bge-reranker-base")

        kimi_key  = os.environ.get("KIMI_API_KEY")
        hyde_llm  = ChatOpenAI(
            model    = retrieval_cfg.get("hyde_model", "moonshot-v1-8k"),
            temperature = 0.3,
            max_tokens  = 400,
            base_url    = "https://api.moonshot.cn/v1",
            api_key     = kimi_key,
        )
        HYDE_PROMPT_R = (
            "你是中国传统命理学专家。请根据以下问题，"
            "仿照古典命理文献（文言文/半文言文）的风格，"
            "写一段 80-150 字的原文片段，直接包含问题答案所涉及的术语和论述。"
            "只输出片段本身，不要标题、序号或解释。\n\n"
            "问题：{question}"
        )

        # 懒加载 BGE 交叉编码器（避免重复初始化）
        _bge_encoder = HuggingFaceCrossEncoder(model_name=rerank_model)

        def _hyde_rerank_retrieve(query) -> list[Document]:
            q = query.get("input", "") if isinstance(query, dict) else query

            # Step 1: HyDE 宽召回
            hyp_texts: list[str] = []
            for _ in range(num_hyp):
                try:
                    hyp = hyde_llm.invoke(HYDE_PROMPT_R.format(question=q)).content.strip()
                    hyp_texts.append(hyp)
                except Exception:
                    hyp_texts.append(q)

            seen: set = set()
            candidates: list[Document] = []
            for hyp_text in hyp_texts:
                for doc in vectorstore.similarity_search(hyp_text, k=hyde_k):
                    key = hash(doc.page_content[:80])
                    if key not in seen:
                        seen.add(key)
                        candidates.append(doc)

            if not candidates:
                return []

            # Step 2: BGE 对原始问题精排（注意：用原始问题，不是假设文档）
            pairs    = [[q, doc.page_content] for doc in candidates]
            scores   = _bge_encoder.score(pairs)
            ranked   = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
            return [doc for _, doc in ranked[:top_n]]

        return RunnableLambda(_hyde_rerank_retrieve)

    # ── RAG-Fusion（多查询 + 倒排融合 RRF）────────────────────────────────────
    elif r_type == "rag_fusion":
        from langchain_classic.retrievers import MergerRetriever
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnableLambda

        num_queries = int(retrieval_cfg.get("num_queries", 3))
        fusion_cfg = {"generation": {**cfg.get("generation", {}), "temperature": 0.3}}
        llm = build_llm(fusion_cfg)

        # 用 LLM 生成多个查询变体，再合并结果
        def multi_query_retriever(question: str):
            expansion_prompt = (
                f"请将以下问题改写成 {num_queries} 个不同角度的中文查询，"
                f"每行一个，不要编号：\n{question}"
            )
            variants_text = (llm | StrOutputParser()).invoke(expansion_prompt)
            variants = [q.strip() for q in variants_text.strip().split("\n") if q.strip()][:num_queries]
            variants.append(question)  # 加上原始查询

            all_docs: list = []
            seen_ids: set = set()
            base = vectorstore.as_retriever(search_kwargs={"k": k})
            for q in variants:
                for doc in base.invoke(q):
                    doc_id = hash(doc.page_content)
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_docs.append(doc)
            # RRF 重排（简化版）
            return all_docs[:k]

        return RunnableLambda(lambda q: multi_query_retriever(q))

    # ── Graph RAG（知识图谱跨书召回 + BGE 精排）─────────────────────────────────
    # 步骤 1：向量宽召回 k 个 seed chunks
    # 步骤 2：在跨书知识图谱中做 hop 跳 BFS，找到来自其他书籍的邻居 chunks
    # 步骤 3：seed + neighbors 合并，BGE 交叉编码器精排，保留 top_n
    # 优势：显式解决 cross_book_hit 问题——即使向量搜索只找到一本书的内容，
    #       图遍历也能通过桥接术语拉取另一本书的相关段落
    elif r_type == "graph_rag":
        import sys
        from pathlib import Path as _Path
        sys.path.insert(0, str(_Path(__file__).parent.parent / "api"))
        from graph_retriever import GraphRetriever, load_graph_and_index

        graph_path       = cfg.get("graph_path", "./data/knowledge_graph.pkl")
        chunk_index_path = cfg.get("chunk_index_path", "./data/chunk_index.json")
        top_n            = int(retrieval_cfg.get("top_n", 7))
        hop              = int(retrieval_cfg.get("hop", 1))
        max_neighbors    = int(retrieval_cfg.get("max_neighbors", 30))
        vector_filter_k  = int(retrieval_cfg.get("vector_filter_k", 0))
        reranker_model   = retrieval_cfg.get("rerank_model", "BAAI/bge-reranker-base")

        G, chunk_index = load_graph_and_index(graph_path, chunk_index_path)

        # 可选 HyDE 种子：在 retrieval 中设 hyde: true 即可启用
        hyde_llm_inst = None
        if retrieval_cfg.get("hyde", False):
            from langchain_openai import ChatOpenAI
            kimi_key = os.environ.get("KIMI_API_KEY")
            hyde_llm_inst = ChatOpenAI(
                model       = retrieval_cfg.get("hyde_model", "moonshot-v1-8k"),
                temperature = 0.3,
                max_tokens  = 400,
                base_url    = "https://api.moonshot.cn/v1",
                api_key     = kimi_key,
            )

        gr = GraphRetriever(
            vectorstore    = vectorstore,
            graph          = G,
            chunk_index    = chunk_index,
            k              = k,
            hop            = hop,
            top_n          = top_n,
            reranker_model = reranker_model,
            max_neighbors  = max_neighbors,
            vector_filter_k = vector_filter_k,
            hyde_llm       = hyde_llm_inst,
        )
        # 预热 BGE 精排器（首次从磁盘加载约 30-40s，之后驻留内存）
        print("    预加载 BGE 精排器（首次约 30~40s）…", flush=True)
        gr._get_reranker()
        print("    BGE 精排器就绪 ✓", flush=True)
        return gr

    else:
        raise ValueError(f"Unknown retrieval type: {r_type}")


# ---------------------------------------------------------------------------
# 单次 RAG 调用（含计时）
# ---------------------------------------------------------------------------


def run_single(question: str, retriever, llm, qa_prompt) -> dict:
    """执行一次检索+生成，返回结果 dict。"""
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    from langchain_classic.chains import create_retrieval_chain

    t0 = time.perf_counter()

    # 检索
    if hasattr(retriever, "invoke"):
        contexts = retriever.invoke(question)
    else:
        contexts = retriever.get_relevant_documents(question)

    t_retrieve = time.perf_counter() - t0

    # 生成
    t1 = time.perf_counter()
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    result = rag_chain.invoke({"input": question, "chat_history": []})
    t_generate = time.perf_counter() - t1

    answer = result.get("answer", "")
    retrieved_contexts = [doc.page_content for doc in result.get("context", contexts)]

    return {
        "question": question,
        "answer": answer,
        "retrieved_contexts": retrieved_contexts,
        "latency_retrieve": round(t_retrieve, 3),
        "latency_generate": round(t_generate, 3),
        "latency_total": round(t_retrieve + t_generate, 3),
    }


# ---------------------------------------------------------------------------
# RAGAS 评估
# ---------------------------------------------------------------------------


def evaluate_with_ragas(records: list[dict], golden_answers: list[str],
                        eval_provider: str | None = None,
                        eval_model: str | None = None) -> dict:
    """
    用 RAGAS 对一批结果评估，返回指标均值 dict。
    若指定 eval_provider/eval_model 则用该 LLM 评估；
    否则优先 KIMI，其次 GROQ，若均不可用则回落到词重叠估算。

    NOTE: RAGAS v0.4.x requires metrics to be instances of ragas.metrics.base.Metric.
    The new ragas.metrics.collections classes inherit from SimpleBaseMetric (NOT Metric),
    so we use the legacy singleton objects from ragas.metrics instead, which pass the
    isinstance(m, Metric) check inside evaluate(). We suppress the DeprecationWarnings
    since the v1.0 migration guide says to use collections — but that API is broken in
    v0.4.3. Embeddings are provided via LangChain's HuggingFaceEmbeddings (no API key).
    """
    try:
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

        from datasets import Dataset
        from ragas import evaluate
        # Use legacy singletons — these ARE ragas.metrics.base.Metric instances.
        # The new ragas.metrics.collections classes fail isinstance(m, Metric) in v0.4.3.
        from ragas.metrics import (  # noqa: deprecated-import
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )
        from ragas.llms import llm_factory, LangchainLLMWrapper
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from openai import OpenAI as OpenAIClient

        from langchain_openai import ChatOpenAI as ChatOpenAILC

        kimi_key = os.environ.get("KIMI_API_KEY")
        google_key = os.environ.get("GOOGLE_API_KEY")
        groq_key = os.environ.get("GROQ_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")

        # ── 指定 eval provider 时优先使用 ─────────────────────────────
        if eval_provider == "openai" and openai_key:
            _model = eval_model or "gpt-4o"
            print(f"  [RAGAS] Using OpenAI ({_model}) for evaluation")
            eval_llm = LangchainLLMWrapper(
                ChatOpenAILC(
                    model=_model,
                    max_tokens=8192,
                    temperature=0,
                    api_key=openai_key,
                )
            )
        elif eval_provider == "kimi" or (eval_provider is None and kimi_key):
            print("  [RAGAS] Using Kimi (moonshot-v1-32k, max_tokens=8192) for evaluation")
            eval_llm = LangchainLLMWrapper(
                ChatOpenAILC(
                    model="moonshot-v1-32k",
                    max_tokens=8192,
                    temperature=0,
                    base_url="https://api.moonshot.cn/v1",
                    api_key=kimi_key,
                )
            )
        elif google_key:
            # Gemini handles RAGAS structured-output (NLI JSON) reliably —
            # much better than Groq/llama which often returns wrong JSON schema.
            print("  [RAGAS] Using Gemini (gemini-2.0-flash) for evaluation")
            from langchain_google_genai import ChatGoogleGenerativeAI
            eval_llm = LangchainLLMWrapper(
                ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
            )
        elif groq_key:
            # Fallback: Groq/llama works but occasionally produces malformed JSON for
            # the faithfulness NLI task; scores may be slightly underestimated.
            print("  [RAGAS] Using Groq (llama-3.3-70b-versatile) for evaluation")
            from langchain_groq import ChatGroq
            eval_llm = LangchainLLMWrapper(
                ChatGroq(model="llama-3.3-70b-versatile", max_tokens=8192, api_key=groq_key)
            )
        else:
            print("  [WARN] No RAGAS-compatible LLM key found, falling back to overlap scoring.")
            return _simple_overlap_scores(records, golden_answers)

        # Local HuggingFace embeddings — no API key required.
        # Use Chinese embedding for answer_relevancy (measures question↔question similarity).
        ragas_emb = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            encode_kwargs={"normalize_embeddings": True},
        )

        # Pre-set llm on each singleton so evaluate() doesn't fall back to OpenAI default.
        faithfulness.llm = eval_llm
        answer_relevancy.llm = eval_llm
        context_recall.llm = eval_llm
        context_precision.llm = eval_llm

        metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

        # Full context — no truncation needed with 32k+ models.
        data = {
            "question": [r["question"] for r in records],
            "answer": [r["answer"] for r in records],
            "contexts": [r["retrieved_contexts"] for r in records],
            "ground_truth": golden_answers,
        }
        ds = Dataset.from_dict(data)
        # Increase timeout: Gemini / remote LLMs sometimes take >60s on first call.
        from ragas import RunConfig
        result = evaluate(
            ds,
            metrics=metrics,
            llm=eval_llm,
            embeddings=ragas_emb,
            run_config=RunConfig(timeout=180, max_retries=3, max_wait=120),
        )
        # EvaluationResult in RAGAS v0.4.x stores per-sample scores in .scores (list of dicts).
        # Compute the mean for each metric manually from .scores.
        all_keys = result.scores[0].keys() if result.scores else []
        import math
        mean_scores = {}
        for k in all_keys:
            vals = [s[k] for s in result.scores if s.get(k) is not None and not (isinstance(s[k], float) and math.isnan(s[k]))]
            mean_scores[k] = round(float(sum(vals) / len(vals)), 4) if vals else float("nan")
        return mean_scores

    except ImportError as e:
        print(f"  [WARN] Missing dependency ({e}), falling back to simple overlap scoring.")
        return _simple_overlap_scores(records, golden_answers)


def _simple_overlap_scores(records: list[dict], golden_answers: list[str]) -> dict:
    """无 RAGAS 时的简单字符重叠估算（仅供参考）。"""
    import re

    def token_set(text: str) -> set:
        return set(re.findall(r"[\w\u4e00-\u9fff]+", text.lower()))

    f1_scores, recall_scores = [], []
    for r, gold in zip(records, golden_answers):
        pred_tokens = token_set(r["answer"])
        gold_tokens = token_set(gold)
        ctx_tokens = token_set(" ".join(r["retrieved_contexts"]))

        if not gold_tokens:
            continue
        recall = len(pred_tokens & gold_tokens) / len(gold_tokens)
        ctx_recall = len(ctx_tokens & gold_tokens) / len(gold_tokens)
        precision = len(pred_tokens & gold_tokens) / max(len(pred_tokens), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        f1_scores.append(f1)
        recall_scores.append(ctx_recall)

    return {
        "answer_f1_overlap": round(sum(f1_scores) / max(len(f1_scores), 1), 4),
        "context_recall_overlap": round(sum(recall_scores) / max(len(recall_scores), 1), 4),
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def run_benchmark(cfg_path: str, dataset: list[dict], output_dir: Path,
                   eval_provider: str | None = None,
                   eval_model: str | None = None) -> dict:
    """跑单个配置的 benchmark，返回汇总指标。"""
    with open(cfg_path, encoding="utf-8") as f:
        cfg: dict = yaml.safe_load(f)
    # Pass eval settings through cfg dict for evaluate_with_ragas
    if eval_provider:
        cfg["_eval_provider"] = eval_provider
    if eval_model:
        cfg["_eval_model"] = eval_model

    cfg_name = Path(cfg_path).stem
    print(f"\n{'='*60}")
    print(f"Config: {cfg_name}  ({cfg_path})")
    print(f"{'='*60}")

    # 初始化 —— 每次 run_benchmark 都强制覆盖 env var，并重置全局缓存，
    # 避免多配置运行时复用上一个配置的 vectorstore / embedding function。
    chroma_dir     = cfg.get("chroma_dir", "./chroma_db")
    embedding_model = cfg.get("embedding_model", "all-MiniLM-L6-v2")

    os.environ["CHROMA_DIR"]      = chroma_dir
    os.environ["EMBEDDING_MODEL"] = embedding_model
    print(f"  ChromaDB   : {chroma_dir}")
    print(f"  Embedding  : {embedding_model}")

    # 重置 chroma_utils 模块级全局缓存（保证新 embedding_model/chroma_dir 生效）
    import chroma_utils as _cu
    _cu._vectorstore       = None
    _cu._embedding_function = None

    from chroma_utils import get_vectorstore
    from fortune_prompts import bench_qa_prompt, bench_qa_prompt_concise, bench_qa_prompt_balanced

    # 根据 YAML 中的 prompt_style 选择 prompt
    prompt_style = cfg.get("prompt_style", "default")
    if prompt_style == "concise":
        qa_prompt = bench_qa_prompt_concise
        print(f"  Prompt: concise (answer_relevancy optimized)")
    elif prompt_style == "balanced":
        qa_prompt = bench_qa_prompt_balanced
        print(f"  Prompt: balanced (relevancy + faithfulness)")
    else:
        qa_prompt = bench_qa_prompt
        print(f"  Prompt: default")

    gen_cfg = cfg.get("generation", {})
    print(f"  LLM: {gen_cfg.get('provider', 'kimi')} / {gen_cfg.get('model', 'moonshot-v1-32k')}")
    llm = build_llm(cfg)

    vs = get_vectorstore()
    retriever = build_retriever(cfg, vs)

    # 批量推理
    records: list[dict] = []
    golden_answers: list[str] = []
    latencies: list[float] = []

    for i, item in enumerate(dataset, 1):
        print(f"  [{i}/{len(dataset)}] Q: {item['question'][:60]}...")
        try:
            result = run_single(item["question"], retriever, llm, qa_prompt)
            records.append(result)
            golden_answers.append(item["golden_answer"])
            latencies.append(result["latency_total"])
            print(f"         latency={result['latency_total']:.2f}s  contexts={len(result['retrieved_contexts'])}")
        except Exception as e:
            print(f"  [ERROR] {e}")

    if not records:
        print("  [ERROR] No records generated, skipping evaluation.")
        return {}

    # 评估
    print("\n  Running RAGAS evaluation...")
    ragas_scores = evaluate_with_ragas(records, golden_answers,
                                         eval_provider=cfg.get("_eval_provider"),
                                         eval_model=cfg.get("_eval_model"))

    # 延迟统计
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    summary = {
        "config": cfg_name,
        "num_samples": len(records),
        "latency_p50": round(latencies_sorted[int(n * 0.5)], 3),
        "latency_p95": round(latencies_sorted[min(int(n * 0.95), n - 1)], 3),
        "latency_mean": round(sum(latencies) / n, 3),
        **ragas_scores,
    }

    # 保存详细结果
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = output_dir / f"{cfg_name}_{timestamp}_detail.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(
            {"config": cfg, "summary": summary, "records": records},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n  Detail saved → {detail_path}")
    print(f"  Summary: {json.dumps(summary, ensure_ascii=False, indent=4)}")
    return summary


def print_comparison_table(summaries: list[dict[str, Any]]) -> None:
    """在终端打印对比表格。"""
    if not summaries:
        return

    # 收集所有指标列
    metric_keys = [k for k in summaries[0].keys() if k not in ("config", "num_samples")]

    # 表头
    col_w = 18
    header = f"{'Config':<20}" + "".join(f"{k:<{col_w}}" for k in metric_keys)
    print("\n" + "=" * len(header))
    print("BENCHMARK COMPARISON")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for s in summaries:
        row = f"{s['config']:<20}" + "".join(f"{str(s.get(k,'')):<{col_w}}" for k in metric_keys)
        print(row)
    print("=" * len(header))


def main():
    # Force UTF-8 line-buffered output for Windows (avoids garbled Chinese in pipes)
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)

    parser = argparse.ArgumentParser(description="RAG Benchmark Runner")
    parser.add_argument("--config", help="Single config YAML file")
    parser.add_argument("--configs", nargs="+", help="Multiple config YAML files")
    parser.add_argument("--dataset", default="benchmarks/qa_dataset.json", help="QA dataset JSON file")
    parser.add_argument("--output-dir", default="benchmarks/results", help="Directory to save results")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of QA samples (for quick runs)")
    parser.add_argument("--eval-provider", default=None, help="Eval LLM provider (openai|kimi|groq)")
    parser.add_argument("--eval-model", default=None, help="Eval LLM model name")
    parser.add_argument("--log", default=None, metavar="FILE",
                        help="Also write all output to FILE (UTF-8). Replaces Tee-Object.")
    args = parser.parse_args()

    # --log: tee stdout to a UTF-8 file (no need for PowerShell Tee-Object)
    if args.log:
        _log_f = open(args.log, "w", encoding="utf-8")
        class _Tee:
            def __init__(self, *streams):
                self._streams = streams
            def write(self, data):
                for s in self._streams:
                    s.write(data)
                    s.flush()
            def flush(self):
                for s in self._streams:
                    s.flush()
        sys.stdout = _Tee(sys.stdout, _log_f)
        sys.stderr = _Tee(sys.stderr, _log_f)

    cfg_paths: list[str] = []
    if args.configs:
        cfg_paths = args.configs
    elif args.config:
        cfg_paths = [args.config]
    else:
        # 默认跑 configs/rag/ 下所有 yaml
        default_dir = Path("configs/rag")
        cfg_paths = [str(p) for p in sorted(default_dir.glob("*.yaml"))]
        if not cfg_paths:
            print("[ERROR] No config files found. Use --config or --configs.")
            sys.exit(1)

    # 加载数据集
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}")
        print("Run scripts/generate_qa_dataset.py first.")
        sys.exit(1)

    with open(dataset_path, encoding="utf-8") as f:
        dataset: list[dict] = json.load(f)

    if args.max_samples:
        dataset = dataset[: args.max_samples]
    print(f"Loaded {len(dataset)} QA samples from {dataset_path}")

    output_dir = Path(args.output_dir)
    summaries: list[dict] = []
    for cfg_path in cfg_paths:
        summary = run_benchmark(cfg_path, dataset, output_dir,
                                eval_provider=args.eval_provider,
                                eval_model=args.eval_model)
        if summary:
            summaries.append(summary)

    # 保存汇总对比
    if len(summaries) > 1:
        compare_path = output_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(compare_path, "w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)
        print(f"\nComparison saved → {compare_path}")
        print_comparison_table(summaries)


if __name__ == "__main__":
    main()
