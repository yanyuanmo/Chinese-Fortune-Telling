"""
rescore_normal_gpt4o.py
~~~~~~~~~~~~~~~~~~~~~~~
Re-evaluate existing normal benchmark results using GPT-4o as the RAGAS eval LLM.
Reads detail JSON files, extracts records, and runs RAGAS evaluation with GPT-4o.

Usage:
    conda activate rag
    python scripts/rescore_normal_gpt4o.py
"""
from __future__ import annotations

import io
import json
import math
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

# ── 要重评的 detail 文件 ──────────────────────────────────────────────────
# (label, detail_json_path)
# Pick representative configs from each category that had full 22Q runs
DETAIL_FILES = [
    # v2 hybrid baselines
    ("hybrid",              "benchmarks/results/v2/hybrid_20260222_150242_detail.json"),
    ("hybrid_rerank_bge",   "benchmarks/results/v2/hybrid_rerank_bge_20260222_151052_detail.json"),
    # v3 proposition-based
    ("prop_vector",         "benchmarks/results/v3/prop_vector_20260222_154732_detail.json"),
    ("prop_hybrid",         "benchmarks/results/v3/prop_hybrid_20260222_155252_detail.json"),
    # v4 hyde baselines
    ("hyde",                "benchmarks/results/v4/hyde_20260222_171757_detail.json"),
    ("hyde_hybrid",         "benchmarks/results/v4/hyde_hybrid_20260222_172541_detail.json"),
    # v5 hyde_rerank series (best variants)
    ("hyde_rerank_topn7",   "benchmarks/results/v5d/hyde_rerank_topn7_20260222_190538_detail.json"),
    ("hyde_rerank_topn9",   "benchmarks/results/v5e/hyde_rerank_topn9_20260222_193129_detail.json"),
    # v8 graph RAG
    ("graph_rag_v3_cot",    "benchmarks/results/v8/graph_rag_v3_cot_only_20260417_161647_detail.json"),
    ("graph_rag_v5_hyde",   "benchmarks/results/v8/graph_rag_v5_hyde_cot_20260417_163224_detail.json"),
    # v9 new runs (answers generated with Kimi, need GPT-4o RAGAS eval)
    ("graph_rag_v7_vf50",   "benchmarks/results/v9_gpt4o/graph_rag_v7_vf50_20260418_104428_detail.json"),
    ("graph_rag_v8_bge_k15","benchmarks/results/v9_gpt4o/graph_rag_v8_bge_base_k15_20260418_105951_detail.json"),
]


def main():
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)

    root = Path(__file__).parent.parent
    os.chdir(root)

    # Load golden answers
    with open("benchmarks/qa_dataset.json", encoding="utf-8") as f:
        dataset = json.load(f)
    golden_map = {item["question"]: item["golden_answer"] for item in dataset}
    print(f"Loaded {len(golden_map)} golden answers from qa_dataset.json")

    # Verify files exist and load records
    configs_data = []
    for label, path in DETAIL_FILES:
        if "SKIP" in path:
            continue
        p = Path(path)
        if not p.exists():
            print(f"  SKIP {label}: {path} not found")
            continue
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        records = data.get("records", [])
        if not records:
            print(f"  SKIP {label}: no records")
            continue
        # Match golden answers
        golden = []
        valid_records = []
        for r in records:
            q = r["question"]
            if q in golden_map:
                golden.append(golden_map[q])
                valid_records.append(r)
        if not valid_records:
            print(f"  SKIP {label}: no matching golden answers")
            continue
        configs_data.append((label, valid_records, golden, data.get("summary", {})))
        print(f"  OK   {label}: {len(valid_records)} records")

    print(f"\nWill re-evaluate {len(configs_data)} configs with GPT-4o RAGAS\n")

    # Import RAGAS components
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")
    from datasets import Dataset
    from ragas import evaluate, RunConfig
    from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
    from ragas.llms import LangchainLLMWrapper
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise EnvironmentError("OPENAI_API_KEY not set")

    eval_llm = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-4o", max_tokens=8192, temperature=0, api_key=openai_key)
    )
    ragas_emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )

    faithfulness.llm = eval_llm
    answer_relevancy.llm = eval_llm
    context_recall.llm = eval_llm
    context_precision.llm = eval_llm

    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

    # Run evaluation for each config
    all_results = []  # (label, old_summary, new_scores)
    for label, records, golden, old_summary in configs_data:
        print(f"  Re-scoring: {label} ({len(records)} questions) ...", flush=True)

        data = {
            "question": [r["question"] for r in records],
            "answer": [r["answer"] for r in records],
            "contexts": [r["retrieved_contexts"] for r in records],
            "ground_truth": golden,
        }
        ds = Dataset.from_dict(data)

        try:
            result = evaluate(
                ds, metrics=metrics, llm=eval_llm, embeddings=ragas_emb,
                run_config=RunConfig(timeout=180, max_retries=3, max_wait=120),
            )
            all_keys = result.scores[0].keys() if result.scores else []
            new_scores = {}
            for k in all_keys:
                vals = [s[k] for s in result.scores
                        if s.get(k) is not None and not (isinstance(s[k], float) and math.isnan(s[k]))]
                new_scores[k] = round(float(sum(vals) / len(vals)), 4) if vals else float("nan")
            print(f"    Done: {new_scores}")
            all_results.append((label, old_summary, new_scores))
        except Exception as e:
            print(f"    ERROR: {e}")

    # Output comparison table
    print("\n" + "=" * 120)
    print("  Normal Benchmark: Kimi eval vs GPT-4o eval")
    print("=" * 120)
    fmt = "  %-26s %8s %8s %8s %8s  |  %8s %8s %8s %8s"
    print(fmt % ("Config", "K_faith", "K_relev", "K_recall", "K_prec",
                 "G_faith", "G_relev", "G_recall", "G_prec"))
    print("  " + "-" * 110)

    rows = []
    for label, old, new in all_results:
        row = {
            "config": label,
            "kimi_faithfulness": old.get("faithfulness", 0),
            "kimi_relevancy": old.get("answer_relevancy", 0),
            "kimi_recall": old.get("context_recall", 0),
            "kimi_precision": old.get("context_precision", 0),
            "gpt4o_faithfulness": new.get("faithfulness", 0),
            "gpt4o_relevancy": new.get("answer_relevancy", 0),
            "gpt4o_recall": new.get("context_recall", 0),
            "gpt4o_precision": new.get("context_precision", 0),
        }
        rows.append(row)
        print("  %-26s %8.3f %8.3f %8.3f %8.3f  |  %8.3f %8.3f %8.3f %8.3f" % (
            label,
            row["kimi_faithfulness"], row["kimi_relevancy"],
            row["kimi_recall"], row["kimi_precision"],
            row["gpt4o_faithfulness"], row["gpt4o_relevancy"],
            row["gpt4o_recall"], row["gpt4o_precision"],
        ))

    print("=" * 120)

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("benchmarks/results") / f"rescore_normal_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved -> {out_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
