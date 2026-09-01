"""Experiment M3: observation-multiplicity sweep for forward-chaining saturation.

Extends Experiment 5 (false-positive amplification) with the multiplicity knob
m in {1, 2, 3}: each retrieved observation is injected as m independent linear
copies. Classic LP (persistent observations) is the m = infinity endpoint.
Two budget regimes: k = 20 (Exp-5 operating point) and k = 200 (budget
non-binding, isolating the effect of m).

Usage (repo root; sell_core.py + multiplicity_core.py required):
    PYTHONHASHSEED=0 python multiplicity_experiment.py
Outputs: results/exp_m3_multiplicity.csv and a console summary with paired
Wilcoxon tests. Sanity-checks m=1 @ k=20 against results/exp5_fp_amplification.csv.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from multiplicity_core import (NeuralScorer, kg_exp5, queries_3hop_comp,  # noqa: E402
                               saturate_classic_lp, saturate_sell)

SEED = 42
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
M_VALUES = [1, 2, 3]
BUDGETS = [20, 200]
OUT = HERE / "results"

rows = []
for eta in NOISE_LEVELS:
    scorer = NeuralScorer(kg_exp5, noise_rate=eta, top_k=20, seed=SEED)
    for inst in queries_3hop_comp:
        obs = scorer.propose_observations(inst)
        gold = inst.answer_entities

        def record(mode, m, budget, goals):
            tp = len(goals & gold)
            fp = len(goals - gold)
            fn = len(gold - goals)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            rows.append({"question_id": inst.question_id, "noise_rate": eta,
                         "mode": mode, "m": m, "budget": budget,
                         "true_positives": tp, "false_positives": fp,
                         "false_negatives": fn, "precision": prec,
                         "recall": rec, "f1": f1})

        derived_lp = saturate_classic_lp(inst.persistent_facts, set(obs),
                                         inst.ground_rules,
                                         goal_predicate="actor_genre_director")
        goals_lp = {a.args[1] for a in derived_lp
                    if a.predicate == "actor_genre_director"}
        record("classic_lp", 0, 0, goals_lp)

        for budget in BUDGETS:
            for m in M_VALUES:
                obs_m = Counter({a: c * m for a, c in obs.items()})
                derived, _ = saturate_sell(inst.persistent_facts, obs_m,
                                           inst.ground_rules, budget=budget,
                                           seed=SEED,
                                           goal_predicate="actor_genre_director")
                goals = {a.args[1] for a in derived
                         if a.predicate == "actor_genre_director"}
                record("sell", m, budget, goals)
    print(f"eta={eta:.1f} done")

df = pd.DataFrame(rows)
OUT.mkdir(exist_ok=True)
df.to_csv(OUT / "exp_m3_multiplicity.csv", index=False)

# ---------------------------------------------------------------- summary
print("\n=== Aggregates (mean per query) ===")
agg = df.groupby(["mode", "m", "budget", "noise_rate"]).agg(
    FP=("false_positives", "mean"), prec=("precision", "mean"),
    rec=("recall", "mean"), f1=("f1", "mean")).reset_index()
print(agg.to_string(index=False, float_format="%.3f"))

# ---------------------------------------------------------------- sanity vs Exp 5
print("\n=== Sanity check vs. exp5_fp_amplification.csv (m=1, k=20 and LP) ===")
e5 = pd.read_csv(OUT / "exp5_fp_amplification.csv")
ok = True
for eta in NOISE_LEVELS:
    for old_mode, cond in [("sell", (df["mode"] == "sell") & (df.m == 1) & (df.budget == 20)),
                           ("classic_lp", df["mode"] == "classic_lp")]:
        a = e5[(e5["mode"] == old_mode) & (e5.noise_rate == eta)].false_positives.mean()
        b = df[cond & (df.noise_rate == eta)].false_positives.mean()
        same = abs(a - b) < 1e-9
        ok &= same
        if not same:
            print(f"  MISMATCH {old_mode} eta={eta}: exp5={a:.3f} vs m-sweep={b:.3f}")
print("  exp5 reproduction:", "EXACT" if ok else "MISMATCH")

# ---------------------------------------------------------------- paired tests
print("\n=== Paired Wilcoxon (k=200, per noise level pooled eta>=0.1) ===")
lp = df[df["mode"] == "classic_lp"].sort_values(["question_id", "noise_rate"])
for m in M_VALUES:
    sm = df[(df["mode"] == "sell") & (df.m == m) & (df.budget == 200)] \
        .sort_values(["question_id", "noise_rate"])
    mask = lp.noise_rate.values >= 0.1
    fp_p = wilcoxon(lp.false_positives.values[mask], sm.false_positives.values[mask]).pvalue \
        if (lp.false_positives.values[mask] != sm.false_positives.values[mask]).any() else float("nan")
    rc_p = wilcoxon(lp.recall.values[mask], sm.recall.values[mask]).pvalue \
        if (lp.recall.values[mask] != sm.recall.values[mask]).any() else float("nan")
    print(f"  m={m}: FP(LP) vs FP(m): p={fp_p:.2e} | recall(LP) vs recall(m): p={rc_p:.2e}")
m1 = df[(df["mode"] == "sell") & (df.m == 1) & (df.budget == 200)].sort_values(["question_id", "noise_rate"])
m2 = df[(df["mode"] == "sell") & (df.m == 2) & (df.budget == 200)].sort_values(["question_id", "noise_rate"])
p = wilcoxon(m2.recall.values, m1.recall.values).pvalue
print(f"  recall gain m=2 vs m=1: mean {m1.recall.mean():.3f} -> {m2.recall.mean():.3f}, p={p:.2e}")
print("DONE")
