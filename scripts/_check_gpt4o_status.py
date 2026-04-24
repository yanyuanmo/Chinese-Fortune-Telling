"""Quick check: GPT-4o eval coverage across all benchmarks."""
import json, math
from pathlib import Path

def is_valid(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v))

# === MULTIHOP ===
print("=== MULTIHOP (36Q, GPT-4o eval) ===")
lb_path = Path("benchmarks/results/multihop/rescore_20260418_102900/leaderboard_merged.json")
lb = json.loads(lb_path.read_text(encoding="utf-8"))
scored = 0
for item in lb:
    cs = item.get("gpt4o_chain_score")
    cfg = item.get("config", "?")
    if is_valid(cs):
        scored += 1
        print(f"  OK   {cfg:35s} gpt4o={cs:.4f}")
    else:
        kimi = item.get("kimi_chain_score", "?")
        print(f"  MISS {cfg:35s} kimi={kimi}")
print(f"  -> {scored}/{len(lb)} scored\n")

# === NORMAL ===
print("=== NORMAL (22Q, GPT-4o eval) ===")
lb2_path = Path("benchmarks/results/rescore_normal_20260418_113757/leaderboard.json")
lb2 = json.loads(lb2_path.read_text(encoding="utf-8"))
scored2 = 0
for item in lb2:
    avg = item.get("gpt4o_avg")
    cfg = item.get("config", "?")
    if is_valid(avg):
        scored2 += 1
        print(f"  OK   {cfg:35s} gpt4o_avg={avg:.4f}")
    else:
        kavg = item.get("kimi_avg")
        print(f"  MISS {cfg:35s} kimi_avg={kavg}")
print(f"  -> {scored2}/{len(lb2)} scored\n")

# === v10 PROMPT VARIANTS ===
print("=== v10 PROMPT VARIANTS ===")
for d in ["v10_concise", "v10_balanced"]:
    dp = Path("benchmarks/results") / d
    if not dp.exists():
        continue
    for f in sorted(dp.glob("*_detail.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        s = data["summary"]
        faith = s.get("faithfulness")
        cfg = s.get("config", "?")
        if is_valid(faith):
            keys = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
            avg = sum(s[k] for k in keys) / 4
            print(f"  OK   {cfg:42s} avg={avg:.4f}")
        else:
            print(f"  NaN  {cfg:42s} (answers saved, RAGAS failed)")
