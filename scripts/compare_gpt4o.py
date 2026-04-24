"""Quick comparison script for GPT-4o eval experiment results."""
import json

# Kimi gen + GPT-4o eval
with open("benchmarks/results/multihop/20260418_092959/results_graph_rag_v3_cot_only.json", encoding="utf-8") as f:
    kimi_gen = json.load(f)

# GPT-4o gen + GPT-4o eval
with open("benchmarks/results/multihop/20260418_095239/results_graph_rag_v3_cot_gpt4o_gen.json", encoding="utf-8") as f:
    gpt_gen = json.load(f)

print(f"{'Q':>3}  {'Kimi gen':>8}  {'GPT4o gen':>9}  {'delta':>6}  note")
print("-" * 45)

better, worse, same = 0, 0, 0
for i, (k, g) in enumerate(zip(kimi_gen, gpt_gen)):
    ks = k["chain_score"]
    gs = g["chain_score"]
    d = gs - ks
    tag = "  <-- GPT4o better" if d > 0.01 else ("  <-- Kimi better" if d < -0.01 else "")
    print(f"{i+1:3d}  {ks:8.3f}  {gs:9.3f}  {d:+6.3f}{tag}")
    if d > 0.01:
        better += 1
    elif d < -0.01:
        worse += 1
    else:
        same += 1

print(f"\nGPT-4o gen better on {better} questions, worse on {worse}, same on {same}")
km = sum(r["chain_score"] for r in kimi_gen) / len(kimi_gen)
gm = sum(r["chain_score"] for r in gpt_gen) / len(gpt_gen)
print(f"Kimi gen  mean chain_score: {km:.4f}")
print(f"GPT4o gen mean chain_score: {gm:.4f}")
print(f"Delta: {gm - km:+.4f}")
