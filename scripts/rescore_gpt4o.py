"""
rescore_gpt4o.py
~~~~~~~~~~~~~~~~
Score-only 模式：复用已有答案文件，用 GPT-4o 重新评分。
不跑检索/生成，只调 GPT-4o evaluate_chain。

用法:
  conda activate rag
  python scripts/rescore_gpt4o.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)

# ── 要重评的结果文件（每个 config 取最新一次完整 36Q 运行） ──────────────
RESULT_FILES = [
    # (label, path)
    ("v2_flat",           "benchmarks/results/multihop/20260223_101647/results_graph_rag_v2.json"),
    ("v3_idf_hop1",       "benchmarks/results/multihop/20260416_225409/results_graph_rag_v3_idf_hop1.json"),
    ("v3_cot",            "benchmarks/results/multihop/20260417_092940/results_graph_rag_v3_cot_only.json"),
    ("v4_cot_v2m3",       "benchmarks/results/multihop/20260417_092940/results_graph_rag_v4_cot.json"),
    ("v5_hyde_cot",       "benchmarks/results/multihop/20260417_153733/results_graph_rag_v5_hyde_cot.json"),
    ("v6_mn10",           "benchmarks/results/multihop/20260417_211506/results_graph_rag_v6_mn10.json"),
    ("v7_vf50",           "benchmarks/results/multihop/20260417_214751/results_graph_rag_v7_vf50.json"),
    ("v7_vf100",          "benchmarks/results/multihop/20260417_214751/results_graph_rag_v7_vf100.json"),
    ("hyde_no_graph",     "benchmarks/results/multihop/20260417_221311/results_hyde_rerank_topn7.json"),
    ("v8_bge_base_k10",   "benchmarks/results/multihop/20260417_222126/results_graph_rag_v8_bge_base.json"),
    ("v8_hyde_bge_base",  "benchmarks/results/multihop/20260417_222126/results_hyde_rerank_v8_bge_base.json"),
    ("v8_reranker_large", "benchmarks/results/multihop/20260417_225002/results_graph_rag_v8_reranker_large.json"),
    ("v8_bge_base_k15",   "benchmarks/results/multihop/20260417_232958/results_graph_rag_v8_bge_base_k15.json"),
    ("v8_bge_base_k20",   "benchmarks/results/multihop/20260417_232958/results_graph_rag_v8_bge_base_k20.json"),
    # ── 追加的 baseline ──
    ("hybrid_feb",        "benchmarks/results/multihop/20260222_204545/results_hybrid.json"),
    ("hyde_old_feb",      "benchmarks/results/multihop/20260222_220558/results_hyde_rerank_topn7.json"),
    ("graph_rag_v1",      "benchmarks/results/multihop/20260222_231229/results_graph_rag.json"),
    ("v3_idf",            "benchmarks/results/multihop/20260416_225409/results_graph_rag_v3_idf.json"),
    ("v6_mn15",           "benchmarks/results/multihop/20260417_211506/results_graph_rag_v6_mn15.json"),
]

EVAL_PROVIDER = "openai"
EVAL_MODEL    = "gpt-4o"
MAX_WORKERS   = 5       # GPT-4o 并发数（评分请求轻量，可以多线程）

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


def evaluate_chain(client, question, reasoning_chain, answer, max_retries=2):
    chain_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(reasoning_chain))
    prompt = CHAIN_EVAL_PROMPT.format(
        question=question, reasoning_chain=chain_str, answer=answer,
    )
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=EVAL_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
                timeout=60,
            )
            raw = resp.choices[0].message.content.strip()
            m = re.search(r"\{[\s\S]*?\}", raw)
            if not m:
                raise ValueError(f"No JSON: {raw[:100]}")
            data = json.loads(m.group())
            scores = data.get("step_scores", [])
            if not scores or not all(s in (0, 1, 0.5) for s in scores):
                raise ValueError(f"Bad step_scores: {scores}")
            scores = scores[:len(reasoning_chain)]
            chain_score = sum(scores) / len(reasoning_chain)
            return round(chain_score, 4)
        except Exception as e:
            if attempt == max_retries:
                return 0.0
            time.sleep(2 * (attempt + 1))


def rescore_one_config(client, label, results):
    """对一个 config 的所有结果重新评分，返回 list[float]。"""
    n = len(results)
    new_scores = [None] * n
    lock = threading.Lock()
    done = [0]

    def _eval(idx):
        r = results[idx]
        s = evaluate_chain(
            client, r["question"], r["reasoning_chain"], r["answer"]
        )
        with lock:
            done[0] += 1
            new_scores[idx] = s
            if done[0] % 6 == 0 or done[0] == n:
                print(f"    {label}: {done[0]}/{n}", flush=True)
        return s

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        list(pool.map(_eval, range(n)))

    return new_scores


def main():
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)

    root = Path(__file__).parent.parent
    os.chdir(root)

    from openai import OpenAI
    # Build eval client
    info = {"openai": {"env_key": "OPENAI_API_KEY", "base_url": None}}
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key, timeout=120.0)
    print(f"Eval LLM: {EVAL_PROVIDER}/{EVAL_MODEL}")

    # --only 过滤
    only_labels = None
    for arg in sys.argv[1:]:
        if arg.startswith("--only="):
            only_labels = set(arg.split("=", 1)[1].split(","))

    # 检查所有文件存在
    configs_data = []
    for label, path in RESULT_FILES:
        if only_labels and label not in only_labels:
            continue
        p = Path(path)
        if not p.exists():
            print(f"  SKIP {label}: {path} not found")
            continue
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if len(data) != 36:
            print(f"  SKIP {label}: only {len(data)} items (need 36)")
            continue
        configs_data.append((label, path, data))

    print(f"\nWill re-score {len(configs_data)} configs × 36 questions = {len(configs_data)*36} evaluations")
    print(f"Estimated time: ~{len(configs_data)*36*8/MAX_WORKERS/60:.0f} minutes\n")

    # 逐 config 重评
    all_results = {}  # label -> {old_scores, new_scores, ...}
    for label, path, data in configs_data:
        print(f"  Re-scoring: {label} ...", flush=True)
        old_scores = [min(r["chain_score"], 1.0) for r in data]
        new_scores = rescore_one_config(client, label, data)

        old_cross = [int(r.get("cross_book_hit", False)) for r in data]
        all_results[label] = {
            "old_scores": old_scores,
            "new_scores": new_scores,
            "cross_hits": old_cross,
            "data": data,
        }

    # ── 输出排行榜 ────────────────────────────────────────────────────────
    print("\n" + "=" * 95)
    print("  GPT-4o Re-scored Leaderboard (all configs, 36 multihop questions)")
    print("=" * 95)
    print(f"  {'Config':<22} {'Kimi eval':>9} {'GPT4o eval':>10} {'Δ':>7} {'hop_ok':>7} {'full_ok':>7} {'cross':>7}")
    print("  " + "-" * 85)

    rows = []
    for label, info in all_results.items():
        old = info["old_scores"]
        new = info["new_scores"]
        cross = info["cross_hits"]
        n = len(new)
        old_mean = sum(old) / n
        new_mean = sum(new) / n
        hop_ok = sum(1 for s in new if s >= 0.6) / n
        full_ok = sum(1 for s in new if s >= 0.99) / n
        cross_rate = sum(cross) / n
        rows.append((label, old_mean, new_mean, new_mean - old_mean, hop_ok, full_ok, cross_rate))

    # Sort by GPT-4o score descending
    rows.sort(key=lambda r: r[2], reverse=True)
    for label, old_m, new_m, delta, hop, full, cross in rows:
        print(f"  {label:<22} {old_m:9.4f} {new_m:10.4f} {delta:+7.3f} {hop:7.1%} {full:7.1%} {cross:7.1%}")

    print("=" * 95)
    best = rows[0]
    print(f"\n  🏆 Best (GPT-4o eval): {best[0]}  chain_score={best[2]:.4f}")

    # 保存完整结果
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("benchmarks/results/multihop") / f"rescore_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存详细分数
    detail = {}
    for label, info in all_results.items():
        detail[label] = {
            "old_scores": info["old_scores"],
            "new_scores": info["new_scores"],
            "cross_hits": info["cross_hits"],
            "questions": [r["question"] for r in info["data"]],
        }
    with open(out_dir / "rescore_detail.json", "w", encoding="utf-8") as f:
        json.dump(detail, f, ensure_ascii=False, indent=2)

    # 保存排行榜
    leaderboard = [
        {"config": r[0], "kimi_eval": round(r[1], 4), "gpt4o_eval": round(r[2], 4),
         "delta": round(r[3], 4), "hop_ok": round(r[4], 4), "full_ok": round(r[5], 4),
         "cross_hit": round(r[6], 4)}
        for r in rows
    ]
    with open(out_dir / "leaderboard.json", "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, ensure_ascii=False, indent=2)

    print(f"\n  Detail → {out_dir / 'rescore_detail.json'}")
    print(f"  Leaderboard → {out_dir / 'leaderboard.json'}")


if __name__ == "__main__":
    main()
