"""
bench_multihop.py
~~~~~~~~~~~~~~~~~
多跳推理 Benchmark：评估 RAG 系统处理跨文档推理的能力。

与普通 rag_bench.py 的区别
─────────────────────────────────────────────────
  评估指标   chain_score = 推理链步骤覆盖率（而非 RAGAS）
  额外统计   cross_book_hit = 检索结果是否覆盖了多个来源书籍
  输入       benchmarks/qa_multihop.json（由 generate_qa_multihop.py 生成）
  输出       benchmarks/results/multihop/<timestamp>/
               results.json     完整逐题得分
               summary.json     汇总对比表
               report.md        可读报告

chain_score 的计算
─────────────────────────────────────────────────
  对每道题：给 Kimi 看推理链 + 模型答案，逐步打分（1/0 分）
  chain_score = sum(step_scores) / len(reasoning_chain)
  额外维度：
    hop_ok_rate   chain_score ≥ 0.6 的题目占比（"基本推对"）
    full_ok_rate  chain_score = 1.0 的题目占比（"完全推对"）
    cross_hit     检索文档覆盖两个来源书籍的占比

用法示例
─────────────────────────────────────────────────
  cd E:\\repos\\Chinese-Fortune-Telling

  # 评估单个配置
  python scripts/bench_multihop.py \\
      --configs configs/rag/v5/hyde_rerank_topn7.yaml \\
      --dataset benchmarks/qa_multihop.json

  # 对比多个配置
  python scripts/bench_multihop.py \\
      --configs configs/rag/v2/hybrid.yaml configs/rag/v5/hyde_rerank_topn7.yaml \\
      --dataset benchmarks/qa_multihop.json \\
      --max-samples 30

  # 仅打分，跳过 RAG（用已有答案文件）
  python scripts/bench_multihop.py --score-only --answers-file benchmarks/results/multihop/.../answers.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent.parent / "api"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from dotenv import load_dotenv

load_dotenv(override=True)

# ── 超参 ──────────────────────────────────────────────────────────────────

EVAL_MODEL        = "moonshot-v1-32k"    # 会被 CLI --eval-model 覆盖
GEN_MODEL         = "moonshot-v1-32k"    # 会被 CLI --gen-model / yaml 覆盖
MAX_WORKERS       = 3
DEFAULT_OUTPUT    = "benchmarks/results/multihop"

# ── Provider 路由 ──────────────────────────────────────────────────────────

PROVIDER_CONFIG = {
    "kimi":    {"env_key": "KIMI_API_KEY",    "base_url": "https://api.moonshot.cn/v1"},
    "openai":  {"env_key": "OPENAI_API_KEY",  "base_url": None},  # 默认 api.openai.com
    "deepseek":{"env_key": "DEEPSEEK_API_KEY", "base_url": "https://api.deepseek.com"},
    "groq":    {"env_key": "GROQ_API_KEY",    "base_url": "https://api.groq.com/openai/v1"},
}


def make_client(provider: str) -> "OpenAI":
    """根据 provider 名称构造 OpenAI-compatible client。"""
    from openai import OpenAI
    info = PROVIDER_CONFIG.get(provider)
    if not info:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(PROVIDER_CONFIG)}")
    api_key = os.environ.get(info["env_key"])
    if not api_key:
        raise EnvironmentError(f"{info['env_key']} not set in .env (required for provider={provider})")
    kwargs = {"api_key": api_key, "timeout": 120.0}
    if info["base_url"]:
        kwargs["base_url"] = info["base_url"]
    return OpenAI(**kwargs)

# ── 评估 Prompt ───────────────────────────────────────────────────────────

CHAIN_EVAL_PROMPT = """\
你是一名中国命理学评卷专家。请评估以下命理多跳推理题的模型答案。

【问题】
{question}

【标准推理链】
{reasoning_chain}

【模型答案】
{answer}

评分标准
─────────────────────
请逐步判断模型答案是否"实质性涵盖"了推理链的每个步骤。
  1 分 = 答案明确提到该步骤的核心命题（允许换表述，但内容等价）
  0 分 = 答案未提及，或内容明显错误

只输出以下 JSON，不加任何其他内容：
{{"step_scores": [1, 0, 1, ...], "chain_score": 0.67, "comment": "..."}}
"""

RAG_ANSWER_PROMPT = """\
你是中国传统命理学研究者，精通《三命通会》《子平真诠》《滴天髓》等古典命理文献。
请严格依据下方提供的古籍原文，用中文回答用户问题。

作答要求（务必逐步推理）：
1. 先分别指出各书中与问题相关的关键原文片段
2. 逐步推导：从每本书的论述出发，说明它们之间的逻辑关联
3. 综合多书观点，得出最终结论
4. 关键术语保留原文用语，用现代汉语阐释
5. 不得编造原文之外的内容

参考古籍原文：
{context}

用户问题：{question}

请按"各书要点 → 逐步推理 → 结论"的结构作答：
"""

# ── 精简版多跳 prompt（从 fortune_prompts.py 引入）──────────────────────────
# 懒加载以避免循环依赖
_RAG_ANSWER_PROMPT_CONCISE = None
_RAG_ANSWER_PROMPT_BALANCED = None

def _get_concise_prompt():
    global _RAG_ANSWER_PROMPT_CONCISE
    if _RAG_ANSWER_PROMPT_CONCISE is None:
        sys.path.insert(0, str(Path(__file__).parent.parent / "api"))
        from fortune_prompts import RAG_ANSWER_PROMPT_CONCISE
        _RAG_ANSWER_PROMPT_CONCISE = RAG_ANSWER_PROMPT_CONCISE
    return _RAG_ANSWER_PROMPT_CONCISE

def _get_balanced_prompt():
    global _RAG_ANSWER_PROMPT_BALANCED
    if _RAG_ANSWER_PROMPT_BALANCED is None:
        sys.path.insert(0, str(Path(__file__).parent.parent / "api"))
        from fortune_prompts import RAG_ANSWER_PROMPT_BALANCED
        _RAG_ANSWER_PROMPT_BALANCED = RAG_ANSWER_PROMPT_BALANCED
    return _RAG_ANSWER_PROMPT_BALANCED


# ─────────────────────────────────────────────────────────────────────────────
# RAG 调用（复用 rag_bench.py 的核心函数）
# ─────────────────────────────────────────────────────────────────────────────

def load_rag_bench():
    """懒加载 rag_bench 的 build_retriever + build_vector_store"""
    bench_dir = Path(__file__).parent
    if str(bench_dir) not in sys.path:
        sys.path.insert(0, str(bench_dir))
    import rag_bench
    return rag_bench


def _with_heartbeat(label: str, fn):
    """Run fn() and print a simple start/done message. No threads (threads
    can propagate KeyboardInterrupt into heavy C-extension calls on Windows)."""
    print(f"  {label}…", flush=True)
    result = fn()
    return result


def build_retriever_from_config(cfg_path: str):
    """从 YAML 配置文件构造检索器，返回 (retriever, config_dict)。

    复用 rag_bench.build_retriever + chroma_utils.get_vectorstore，
    与 run_benchmark() 中的初始化逻辑保持一致。
    """
    rb = load_rag_bench()

    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    chroma_dir      = cfg.get("chroma_dir", "./chroma_db_bge")
    embedding_model = cfg.get("embedding_model", "BAAI/bge-small-zh-v1.5")
    os.environ["CHROMA_DIR"]      = chroma_dir
    os.environ["EMBEDDING_MODEL"] = embedding_model

    # 重置 chroma_utils 全局缓存，保证切换配置时不复用旧 vectorstore
    import chroma_utils as _cu
    _cu._vectorstore        = None
    _cu._embedding_function = None

    from chroma_utils import get_vectorstore
    vs = _with_heartbeat("[2/3] 加载 embedding 模型", get_vectorstore)
    print(f"\r  [2/3] 向量存储就绪 ✓ (count={vs._collection.count()})        ", flush=True)

    retriever = _with_heartbeat("[3/3] 构建检索器（首次加载 BGE 模型）", lambda: rb.build_retriever(cfg, vs))
    print(f"\r  [3/3] 检索器就绪 ✓                                           ", flush=True)
    return retriever, cfg


def retrieve_and_answer(
    question: str,
    retriever,
    gen_client,
    gen_model: str = None,
    prompt_style: str = "default",
) -> tuple[str, list[dict]]:
    """用检索器拉上下文，调 LLM 生成答案。返回 (answer, docs)"""
    gen_model = gen_model or GEN_MODEL
    docs = []
    for attempt in range(3):
        try:
            docs = retriever.invoke(question)
            break
        except Exception as e:
            if attempt == 2:
                # 检索彻底失败，返回空
                return f"[retrieve_error: {e}]", []
            time.sleep(3 * (attempt + 1))
            try:
                docs = retriever.get_relevant_documents(question)
                break
            except Exception:
                time.sleep(3 * (attempt + 1))

    context_parts = []
    doc_metas = []
    for doc in docs:
        context_parts.append(doc.page_content)
        doc_metas.append({
            "content": doc.page_content[:200],
            "book": (doc.metadata or {}).get("book", ""),
        })

    context = "\n\n---\n\n".join(context_parts)

    if prompt_style == "concise":
        prompt_template = _get_concise_prompt()
    elif prompt_style == "balanced":
        prompt_template = _get_balanced_prompt()
    else:
        prompt_template = RAG_ANSWER_PROMPT
    prompt = prompt_template.format(context=context, question=question)

    answer = ""
    for _attempt in range(3):
        try:
            resp = gen_client.chat.completions.create(
                model=gen_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800,
                timeout=90,
            )
            answer = resp.choices[0].message.content.strip()
            break
        except Exception as e:
            if _attempt < 2 and ("429" in str(e) or "overloaded" in str(e)):
                time.sleep(5 * (_attempt + 1))
                continue
            answer = f"[api_error: {e}]"
    return answer, doc_metas


# ─────────────────────────────────────────────────────────────────────────────
# Chain-score 评估
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_chain(
    question: str,
    reasoning_chain: list[str],
    answer: str,
    eval_client,
    eval_model: str = None,
    max_retries: int = 2,
) -> dict:
    """对单道题打 chain_score。返回 {step_scores, chain_score, comment}"""
    eval_model = eval_model or EVAL_MODEL
    chain_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(reasoning_chain))
    prompt = CHAIN_EVAL_PROMPT.format(
        question=question,
        reasoning_chain=chain_str,
        answer=answer,
    )
    for attempt in range(max_retries + 1):
        try:
            resp = eval_client.chat.completions.create(
                model=eval_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
                timeout=60,
            )
            raw = resp.choices[0].message.content.strip()
            m = re.search(r"\{[\s\S]*?\}", raw)
            if not m:
                raise ValueError(f"No JSON found: {raw[:100]}")
            data = json.loads(m.group())
            scores = data.get("step_scores", [])
            # 校验
            if not scores or not all(s in (0, 1, 0.5) for s in scores):
                raise ValueError(f"Bad step_scores: {scores}")
            # LLM 可能返回多于 reasoning_chain 长度的评分，截断以防 chain_score > 1
            scores = scores[:len(reasoning_chain)]
            chain_score = sum(scores) / len(reasoning_chain)
            return {
                "step_scores":  scores,
                "chain_score":  round(chain_score, 4),
                "comment":      data.get("comment", ""),
            }
        except Exception as e:
            if attempt == max_retries:
                return {"step_scores": [], "chain_score": 0.0, "comment": f"eval_error: {e}"}
            time.sleep(2)


# ─────────────────────────────────────────────────────────────────────────────
# 单题处理
# ─────────────────────────────────────────────────────────────────────────────

def run_single_item(
    item: dict,
    retriever,
    gen_client,
    eval_client,
    cfg_name: str,
    gen_model: str = None,
    eval_model: str = None,
    prompt_style: str = "default",
) -> dict:
    """执行单题：检索 + 生成 + chain_score 评估"""
    t0 = time.perf_counter()
    question        = item["question"]
    reasoning_chain = item["reasoning_chain"]
    required_hops   = item.get("required_hops", len(reasoning_chain))
    source_books    = {item["metadata"]["book1"], item["metadata"]["book2"]}

    try:
        answer, doc_metas = retrieve_and_answer(question, retriever, gen_client, gen_model, prompt_style)
    except Exception as e:
        answer = f"[retrieve_fatal: {e}]"
        doc_metas = []
    t_retrieve = time.perf_counter() - t0

    # cross_book_hit: 检索到的文档是否覆盖了两个来源书籍
    retrieved_books = {d["book"] for d in doc_metas if d["book"]}
    cross_hit = len(source_books & retrieved_books) == len(source_books)

    # chain-score
    eval_result = evaluate_chain(question, reasoning_chain, answer, eval_client, eval_model)

    total_time = time.perf_counter() - t0
    return {
        "id":             item.get("id", ""),
        "config":         cfg_name,
        "question":       question,
        "answer":         answer,
        "golden_answer":  item.get("golden_answer", ""),
        "reasoning_chain": reasoning_chain,
        "step_scores":    eval_result["step_scores"],
        "chain_score":    eval_result["chain_score"],
        "comment":        eval_result["comment"],
        "required_hops":  required_hops,
        "cross_book_hit": cross_hit,
        "retrieved_books": list(retrieved_books),
        "source_books":   list(source_books),
        "metadata":       item.get("metadata", {}),
        "latency_s":      round(total_time, 2),
        "retrieve_s":     round(t_retrieve, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 汇总统计
# ─────────────────────────────────────────────────────────────────────────────

def summarize_results(results: list[dict], cfg_name: str) -> dict:
    if not results:
        return {"config": cfg_name, "n": 0}

    n               = len(results)
    chain_scores    = [r["chain_score"] for r in results]
    cross_hits      = [r["cross_book_hit"] for r in results]
    latencies       = sorted(r["latency_s"] for r in results)

    hop_ok          = sum(1 for s in chain_scores if s >= 0.6)
    full_ok         = sum(1 for s in chain_scores if s >= 0.99)

    # 按 required_hops 分组
    by_hops: dict[int, list[float]] = {}
    for r in results:
        h = r.get("required_hops", 0)
        by_hops.setdefault(h, []).append(r["chain_score"])

    by_hops_avg = {k: round(sum(v) / len(v), 4) for k, v in sorted(by_hops.items())}

    # 按书籍对分组
    by_pair: dict[str, list[float]] = {}
    for r in results:
        key = f"{r['metadata'].get('book1','?')}×{r['metadata'].get('book2','?')}"
        by_pair.setdefault(key, []).append(r["chain_score"])

    by_pair_avg = {k: round(sum(v) / len(v), 4) for k, v in sorted(by_pair.items())}

    return {
        "config":        cfg_name,
        "n":             n,
        "chain_score_mean":  round(sum(chain_scores) / n, 4),
        "chain_score_min":   round(min(chain_scores), 4),
        "chain_score_max":   round(max(chain_scores), 4),
        "hop_ok_rate":       round(hop_ok / n, 4),   # chain_score ≥ 0.6
        "full_ok_rate":      round(full_ok / n, 4),  # chain_score = 1.0
        "cross_book_hit_rate": round(sum(cross_hits) / n, 4),
        "latency_p50":       round(latencies[n // 2], 2),
        "latency_p90":       round(latencies[int(n * 0.9)], 2),
        "by_required_hops":  by_hops_avg,
        "by_book_pair":      by_pair_avg,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 对比表打印
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison(summaries: list[dict]) -> None:
    if not summaries:
        return

    cols = [
        ("Config",         "config",             20),
        ("N",              "n",                   4),
        ("chain_mean",     "chain_score_mean",    10),
        ("hop_ok≥0.6",     "hop_ok_rate",         10),
        ("full_ok=1.0",    "full_ok_rate",        10),
        ("cross_hit",      "cross_book_hit_rate", 10),
        ("p50_lat",        "latency_p50",          8),
    ]

    header = "  ".join(f"{h:<{w}}" for h, _, w in cols)
    sep    = "  ".join("-" * w for _, _, w in cols)
    print("\n" + "=" * len(header))
    print("  Multi-hop RAG Benchmark Results")
    print("=" * len(header))
    print(header)
    print(sep)

    for s in summaries:
        row = "  ".join(
            f"{str(s.get(k, '')):<{w}}"
            for _, k, w in cols
        )
        print(row)

    print("=" * len(header))

    # 详细：按 required_hops 细分
    print("\n  chain_score by required_hops:")
    all_hop_keys = sorted({
        hk
        for s in summaries
        for hk in s.get("by_required_hops", {})
    })
    hop_header = f"  {'Config':<20}" + "".join(f"  {h}hop" for h in all_hop_keys)
    print(hop_header)
    for s in summaries:
        hop_vals = "".join(
            f"  {s['by_required_hops'].get(h, 'N/A'):<5}"
            for h in all_hop_keys
        )
        print(f"  {s['config']:<20}{hop_vals}")


# ─────────────────────────────────────────────────────────────────────────────
# Markdown 报告
# ─────────────────────────────────────────────────────────────────────────────

def write_report(
    summaries: list[dict],
    all_results: dict[str, list[dict]],
    output_dir: Path,
    dataset_path: str,
) -> None:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Multi-hop RAG Benchmark Report",
        f"\n生成时间：{now_str}",
        f"\n数据集：{dataset_path}",
        f"\n样本数：{summaries[0]['n'] if summaries else 0}",
        "\n## 汇总对比\n",
        "| Config | chain_mean | hop_ok≥0.6 | full_ok=1.0 | cross_hit | p50 lat |",
        "|--------|-----------|-----------|------------|----------|---------|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['config']} | {s['chain_score_mean']} | {s['hop_ok_rate']}"
            f" | {s['full_ok_rate']} | {s['cross_book_hit_rate']} | {s['latency_p50']}s |"
        )

    lines.append("\n## chain_score 按推理步骤数细分\n")
    all_hop_keys = sorted({hk for s in summaries for hk in s.get("by_required_hops", {})})
    hop_header = "| Config | " + " | ".join(f"{h}hop" for h in all_hop_keys) + " |"
    lines.append(hop_header)
    lines.append("|" + "--------|" * (len(all_hop_keys) + 1))
    for s in summaries:
        vals = " | ".join(str(s["by_required_hops"].get(h, "N/A")) for h in all_hop_keys)
        lines.append(f"| {s['config']} | {vals} |")

    lines.append("\n## chain_score 按书籍对细分\n")
    all_pair_keys = sorted({pk for s in summaries for pk in s.get("by_book_pair", {})})
    if all_pair_keys:
        pair_header = "| Config | " + " | ".join(all_pair_keys) + " |"
        lines.append(pair_header)
        lines.append("|" + "--------|" * (len(all_pair_keys) + 1))
        for s in summaries:
            pvals = " | ".join(str(s["by_book_pair"].get(p, "N/A")) for p in all_pair_keys)
            lines.append(f"| {s['config']} | {pvals} |")

    # 底部追加逐题详情（第一个配置）
    if all_results:
        cfg0 = list(all_results.keys())[0]
        lines.append(f"\n## 逐题得分（{cfg0}）\n")
        lines.append("| # | chain | hops | books | question (truncated) |")
        lines.append("|---|-------|------|-------|----------------------|")
        for i, r in enumerate(all_results[cfg0], 1):
            lines.append(
                f"| {i} | {r['chain_score']} | {r['required_hops']}"
                f" | {r['metadata'].get('book1','?')}×{r['metadata'].get('book2','?')}"
                f" | {r['question'][:40]}… |"
            )

    report_path = output_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Report written → {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Force UTF-8 line-buffered output for Windows (avoids garbled Chinese in pipes)
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)

    parser = argparse.ArgumentParser(description="Multi-hop RAG Benchmark")
    parser.add_argument(
        "--configs", nargs="+", required=True,
        help="YAML config files for RAG systems to compare",
    )
    parser.add_argument(
        "--dataset", default="benchmarks/qa_multihop.json",
        help="Multi-hop QA dataset (JSON)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit number of questions (for quick tests)",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT,
        help="Directory to write result files",
    )
    parser.add_argument(
        "--no-parallel", action="store_true",
        help="Disable concurrent evaluation (easier debugging)",
    )
    parser.add_argument(
        "--eval-provider", default="openai",
        help="Provider for evaluation LLM (openai/kimi/deepseek/groq). Default: openai",
    )
    parser.add_argument(
        "--eval-model", default="gpt-4o",
        help="Model name for evaluation. Default: gpt-4o",
    )
    parser.add_argument(
        "--gen-provider", default=None,
        help="Provider for generation LLM. Default: from yaml config or kimi",
    )
    parser.add_argument(
        "--gen-model", default=None,
        help="Model name for generation. Default: from yaml config or moonshot-v1-32k",
    )
    parser.add_argument(
        "--log", default=None, metavar="FILE",
        help="Also write all output to FILE (UTF-8). Replaces Tee-Object.",
    )
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

    root = Path(__file__).parent.parent
    os.chdir(root)

    # ── API clients ────────────────────────────────────────────────────────
    global EVAL_MODEL, GEN_MODEL

    # Eval client (默认 GPT-4o，避免自评偏差)
    EVAL_MODEL = args.eval_model
    eval_client = make_client(args.eval_provider)
    print(f"  Eval  LLM: {args.eval_provider}/{EVAL_MODEL}")

    # Gen client (默认从 yaml 读取，CLI 可覆盖)
    gen_provider = args.gen_provider or "kimi"
    GEN_MODEL    = args.gen_model or "moonshot-v1-32k"
    gen_client   = make_client(gen_provider)
    print(f"  Gen   LLM: {gen_provider}/{GEN_MODEL}")

    # ── 加载数据集 ─────────────────────────────────────────────────────────
    dataset_path = args.dataset
    with open(dataset_path, encoding="utf-8") as f:
        dataset: list[dict] = json.load(f)

    if args.max_samples:
        dataset = dataset[: args.max_samples]

    print(f"Loaded {len(dataset)} multi-hop QA samples from {dataset_path}")
    hop_dist: dict[int, int] = {}
    for item in dataset:
        h = item.get("required_hops", len(item.get("reasoning_chain", [])))
        hop_dist[h] = hop_dist.get(h, 0) + 1
    print(f"  required_hops distribution: { {k: hop_dist[k] for k in sorted(hop_dist)} }")

    # ── 创建输出目录 ───────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output → {out_dir}")

    # ── 逐配置运行 ─────────────────────────────────────────────────────────
    all_results:  dict[str, list[dict]] = {}
    all_summaries: list[dict]           = []

    for cfg_path in args.configs:
        cfg_name = Path(cfg_path).stem
        print(f"\n{'═'*60}")
        print(f"  Config: {cfg_name}  ({cfg_path})")
        print(f"{'═'*60}")

        try:
            retriever, cfg = build_retriever_from_config(cfg_path)
        except Exception as exc:
            import traceback as _tb
            print(f"  ERROR building retriever: {exc}")
            _tb.print_exc()
            continue

        # 如果 CLI 没有指定 gen-provider/gen-model, 尝试从 yaml 中读取
        cfg_gen      = cfg.get("generation", {})
        cur_gen_prov = args.gen_provider or cfg_gen.get("provider", "kimi")
        cur_gen_mod  = args.gen_model    or cfg_gen.get("model", "moonshot-v1-32k")
        # 如果本配置的 gen_provider 和全局不同, 构建新 client
        if cur_gen_prov != gen_provider:
            cur_gen_client = make_client(cur_gen_prov)
            print(f"  (Config overrides gen → {cur_gen_prov}/{cur_gen_mod})")
        else:
            cur_gen_client = gen_client
            cur_gen_mod    = cur_gen_mod if cur_gen_mod != GEN_MODEL else GEN_MODEL

        # prompt_style 从 YAML 配置中读取
        prompt_style = cfg.get("prompt_style", "default")
        if prompt_style != "default":
            print(f"  Prompt: {prompt_style}")

        results: list[dict] = []

        workers = 1 if args.no_parallel else MAX_WORKERS

        if workers == 1:
            checkpoint_file = out_dir / f"checkpoint_{cfg_name}.jsonl"
            _ckpt_seq = open(checkpoint_file, "w", encoding="utf-8")
            # 忽略 SIGINT（Ctrl+C / Windows 遗留信号），防止 KI 打断单题循环
            import signal as _signal
            _orig_sigint = _signal.signal(_signal.SIGINT, _signal.SIG_IGN)
            for i, item in enumerate(dataset, 1):
                try:
                    r = run_single_item(item, retriever, cur_gen_client, eval_client, cfg_name, cur_gen_mod, EVAL_MODEL, prompt_style)
                except (Exception, KeyboardInterrupt) as exc:
                    import traceback as _tb
                    print(f"\n  ERROR on item {i}: {type(exc).__name__}: {exc}", flush=True)
                    if not isinstance(exc, KeyboardInterrupt):
                        _tb.print_exc()
                    r = {
                        "id": item.get("id", ""), "config": cfg_name,
                        "question": item["question"], "answer": f"[fatal: {exc}]",
                        "golden_answer": item.get("golden_answer", ""),
                        "reasoning_chain": item["reasoning_chain"],
                        "step_scores": [], "chain_score": 0.0,
                        "comment": f"fatal_error: {type(exc).__name__}",
                        "required_hops": item.get("required_hops", 3),
                        "cross_book_hit": False, "retrieved_books": [],
                        "source_books": [item["metadata"]["book1"], item["metadata"]["book2"]],
                        "metadata": item.get("metadata", {}),
                        "latency_s": 0.0, "retrieve_s": 0.0,
                    }
                results.append(r)
                _ckpt_seq.write(json.dumps(r, ensure_ascii=False) + "\n")
                _ckpt_seq.flush()
                print(
                    f"  [{i:3d}/{len(dataset)}] chain={r['chain_score']:.3f}"
                    f"  cross={int(r['cross_book_hit'])}  lat={r['latency_s']:.1f}s"
                    f"  | {r['question'][:30]}…",
                    flush=True,
                )
            _ckpt_seq.close()
            _signal.signal(_signal.SIGINT, _orig_sigint)  # 恢复信号处理
        else:
            lock = threading.Lock()
            done = [0]
            checkpoint_file = out_dir / f"checkpoint_{cfg_name}.jsonl"
            _ckpt_f = open(checkpoint_file, "a", encoding="utf-8")

            def _run(item):
                try:
                    r = run_single_item(item, retriever, cur_gen_client, eval_client, cfg_name, cur_gen_mod, EVAL_MODEL, prompt_style)
                except Exception as exc:
                    r = {
                        "id":             item.get("id", ""),
                        "config":         cfg_name,
                        "question":       item["question"],
                        "answer":         f"[fatal: {exc}]",
                        "golden_answer":  item.get("golden_answer", ""),
                        "reasoning_chain": item["reasoning_chain"],
                        "step_scores":    [],
                        "chain_score":    0.0,
                        "comment":        f"fatal_error: {exc}",
                        "required_hops":  item.get("required_hops", 3),
                        "cross_book_hit": False,
                        "retrieved_books": [],
                        "source_books":   [item["metadata"]["book1"], item["metadata"]["book2"]],
                        "metadata":       item.get("metadata", {}),
                        "latency_s":      0.0,
                        "retrieve_s":     0.0,
                    }
                with lock:
                    done[0] += 1
                    print(
                        f"  [{done[0]:3d}/{len(dataset)}] chain={r['chain_score']:.3f}"
                        f"  cross={int(r['cross_book_hit'])}  lat={r['latency_s']:.1f}s"
                        f"  | {r['question'][:30]}…"
                    )
                    _ckpt_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    _ckpt_f.flush()
                return r

            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_run, dataset))
            _ckpt_f.close()

        all_results[cfg_name] = results

        # 保存逐题结果
        results_file = out_dir / f"results_{cfg_name}.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        summary = summarize_results(results, cfg_name)
        all_summaries.append(summary)
        print(
            f"\n  chain_score: mean={summary['chain_score_mean']}  "
            f"hop_ok≥0.6={summary['hop_ok_rate']}  "
            f"full_ok=1.0={summary['full_ok_rate']}\n"
            f"  cross_hit={summary['cross_book_hit_rate']}  "
            f"p50={summary['latency_p50']}s"
        )

    # ── 汇总 & 报告 ────────────────────────────────────────────────────────
    print_comparison(all_summaries)

    # 保存汇总 JSON
    summary_file = out_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)
    print(f"\n  Summary → {summary_file}")

    # Markdown 报告
    write_report(all_summaries, all_results, out_dir, dataset_path)

    # ── 解读提示 ───────────────────────────────────────────────────────────
    if all_summaries:
        best = max(all_summaries, key=lambda s: s["chain_score_mean"])
        print(f"\n🏆  Best config: {best['config']}  chain_score={best['chain_score_mean']}")

    print(
        f"\n💡  Graph RAG 改进目标：chain_score 比最佳 Flat RAG baseline 提升 ≥ 0.15\n"
        f"    （尤其是 required_hops≥3 的题目）\n"
    )


if __name__ == "__main__":
    import traceback as _root_tb
    try:
        main()
    except BaseException as _root_exc:
        with open("bench_crash_log.txt", "w", encoding="utf-8") as _cf:
            _root_tb.print_exc(file=_cf)
            _cf.write(f"\nCrash type: {type(_root_exc).__name__}: {_root_exc}\n")
        _root_tb.print_exc()
        raise
