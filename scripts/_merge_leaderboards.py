import json

with open("benchmarks/results/multihop/rescore_20260418_101844/leaderboard.json", encoding="utf-8") as f:
    lb1 = json.load(f)
with open("benchmarks/results/multihop/rescore_20260418_102900/leaderboard.json", encoding="utf-8") as f:
    lb2 = json.load(f)

all_rows = lb1 + lb2
all_rows.sort(key=lambda r: r["gpt4o_eval"], reverse=True)

new_labels = {"hybrid_feb", "hyde_old_feb", "graph_rag_v1", "v3_idf", "v6_mn15"}

print("=" * 100)
print("  COMPLETE GPT-4o Leaderboard (19 configs, 36 multihop questions)")
print("=" * 100)
fmt = "  %2s %-26s %9s %10s %7s %7s %7s %7s"
print(fmt % ("#", "Config", "Kimi", "GPT4o", "delta", "hop_ok", "full_ok", "cross"))
print("  " + "-" * 92)
for i, r in enumerate(all_rows, 1):
    tag = " *NEW*" if r["config"] in new_labels else ""
    name = r["config"] + tag
    print(
        "  %2d %-26s %9.4f %10.4f %+7.3f %6.1f%% %6.1f%% %6.1f%%"
        % (
            i, name, r["kimi_eval"], r["gpt4o_eval"], r["delta"],
            r["hop_ok"] * 100, r["full_ok"] * 100, r["cross_hit"] * 100,
        )
    )
print("=" * 100)

with open(
    "benchmarks/results/multihop/rescore_20260418_102900/leaderboard_merged.json",
    "w", encoding="utf-8",
) as f:
    json.dump(all_rows, f, ensure_ascii=False, indent=2)
print("\nSaved -> leaderboard_merged.json")
