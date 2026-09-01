"""Experiment M2: trained retriever front-end for SELL on MetaQA.

A staged, *trained* retriever in the spirit of query-graph/subgraph KGQA systems:
1. A relation-path classifier (TF-IDF + logistic regression, one per hop level)
   is trained on the MetaQA *training* split, with path labels obtained by the
   same answer-set path recovery used for the test protocol.
2. At test time the classifier predicts the relation path from the (entity-
   masked) question, and observations are proposed by hop-ordered subgraph
   expansion from the topic entity along the predicted path, capped at K.

Also verifies the Supercondriaque duplicate-use case under multiplicity m=2.

Usage (repo root; requires metaqa_experiment.py + sell_core.py):
    PYTHONHASHSEED=0 python metaqa_trained_experiment.py
Outputs: results/exp_m2_trained.csv, results/exp_m2_summary.json.
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Re-execute the M1 setup (KB, path recovery, instance building, corpora)
# byte-identically, stopping before its experiment loops.
_MARKER = "# ------------------------------------------------------- experiment loops"
_setup = (HERE / "metaqa_experiment.py").read_text().split(_MARKER)[0]
exec(compile(_setup, "metaqa_experiment.py[setup]", "exec"))

from sell_core import run_neural_only, run_sell_pipeline, verify_trace  # noqa: E402

N_TRAIN = 3000
K_VALUES_M2 = [50, 100, 200]
K_MAIN_M2 = 100

# ---------------------------------------------------------------- training data
def mask_topic(text, topic):
    return text.replace(topic, " topicentity ") if topic in text \
        else re.sub(re.escape(topic), " topicentity ", text, flags=re.I)


def path_label(path):
    return "|".join(f"{r}:{d}" for r, d in path)


print("Building path-classifier training data from MetaQA train split ...")
train_files = {h: DATA / f"{h}-hop" / "qa_train.txt" for h in (1, 2, 3)}
for h, f in train_files.items():
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{METAQA_MIRROR}/{h}-hop/vanilla/qa_train.txt", f)

classifiers = {}
for hops in (1, 2, 3):
    rng = random.Random(SEED + 100 + hops)
    lines = train_files[hops].read_text().rstrip("\n").split("\n")
    rng.shuffle(lines)
    X_txt, y = [], []
    for line in lines:
        if len(X_txt) >= N_TRAIN:
            break
        q, ans = line.split("\t")
        m = re.search(r"\[(.+?)\]", q)
        if not m:
            continue
        topic = lower2canon.get(m.group(1).lower())
        if topic is None:
            continue
        gold = set()
        ok = True
        for a in ans.split("|"):
            c = lower2canon.get(a.lower())
            if c is None:
                ok = False
                break
            gold.add(c)
        if not ok or not gold:
            continue
        path = recover_path(topic, gold, hops)
        if path is None:
            continue
        X_txt.append(mask_topic(q.replace("[", "").replace("]", ""), topic))
        y.append(path_label(path))
    vec = TfidfVectorizer(analyzer="word", lowercase=True, ngram_range=(1, 2))
    Xv = vec.fit_transform(X_txt)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xv, y)
    classifiers[hops] = (vec, clf)
    print(f"  {hops}-hop: {len(X_txt)} training questions, "
          f"{len(set(y))} path classes, train acc = {clf.score(Xv, y):.4f}")

# ---------------------------------------------------------------- test-time accuracy
print("Path-classifier accuracy on the admitted test questions:")
clf_acc = {}
for hops, qs in queries.items():
    vec, clf = classifiers[hops]
    correct = 0
    for q in qs:
        true_path = recover_path(q.topic_entity, q.answer_entities, q.hops)
        pred = clf.predict(vec.transform([mask_topic(q.question_text, q.topic_entity)]))[0]
        correct += int(pred == path_label(true_path))
    clf_acc[hops] = correct / len(qs)
    print(f"  {hops}-hop: {clf_acc[hops]:.3f} ({correct}/{len(qs)})")


# ---------------------------------------------------------------- trained retriever
class TrainedPathRetriever:
    """Predicted-path, hop-ordered subgraph expansion capped at K."""

    def __init__(self, classifiers, top_k):
        self.classifiers = classifiers
        self.top_k = top_k
        self._cache = {}

    def propose_observations(self, instance):
        key = instance.question_id
        if key not in self._cache:
            vec, clf = self.classifiers[instance.hops]
            masked = mask_topic(instance.question_text, instance.topic_entity)
            pred = clf.predict(vec.transform([masked]))[0]
            path = [tuple(p.split(":")) for p in pred.split("|")]
            obs = Counter()
            frontier = {instance.topic_entity}
            for i, (r, d) in enumerate(path):
                hop_atoms, nxt = [], set()
                for node in sorted(frontier):
                    for x in sorted(step(node, r, d)):
                        if i < len(path) - 1 and x == instance.topic_entity:
                            continue  # MetaQA semantics: no topic revisit
                        hop_atoms.append(edge_atom(node, x, r, d))
                        nxt.add(x)
                for a in hop_atoms:  # earlier hops keep priority under the K cap
                    if len(obs) >= self.top_k:
                        break
                    obs[a] = 1
                frontier = nxt
            self._cache[key] = obs
        return Counter(self._cache[key])


# ---------------------------------------------------------------- experiment
rows, perquery = [], []
verify_total = verify_pass = budget_viol = 0
for K in K_VALUES_M2:
    retr = TrainedPathRetriever(classifiers, top_k=K)
    for hops, qs in queries.items():
        recalls = [len(set(retr.propose_observations(q)) & q.obs_required_facts)
                   / len(q.obs_required_facts) for q in qs if q.obs_required_facts]
        for mode in ("sell", "classic_lp", "neural_only"):
            t0 = time.time()
            h1l, f1l, prl, ovf, ntr = [], [], [], 0, 0
            for q in qs:
                res = run_neural_only(q, retr) if mode == "neural_only" \
                    else run_sell_pipeline(q, retr, mode=mode, budget=BUDGET)
                h1l.append(1.0 if res["hits_at_1"] else 0.0)
                f1l.append(res["f1"])
                prl.append(1.0 if res["proved"] else 0.0)
                for tr in res.get("traces", []):
                    ntr += 1
                    if tr.budget_used > BUDGET:
                        ovf += 1
                if mode == "sell" and K == K_MAIN_M2:
                    perquery.append({"question_id": q.question_id, "hops": hops,
                                     "hits_at_1": res["hits_at_1"], "f1": res["f1"],
                                     "proved": res["proved"]})
                    for tr in res["traces"]:
                        verify_total += 1
                        acc, _ = verify_trace(tr, q.persistent_facts,
                                              Counter(retr.propose_observations(q)), BUDGET)
                        verify_pass += int(acc)
                        budget_viol += int(tr.budget_used > BUDGET)
            rows.append({"retriever": "trained_path", "K": K, "hops": hops, "mode": mode,
                         "hits1": float(np.mean(h1l)), "f1": float(np.mean(f1l)),
                         "proof_rate": float(np.mean(prl)), "obs_recall": float(np.mean(recalls)),
                         "budget_overflow": ovf, "total_traces": ntr,
                         "n_queries": len(qs), "time_s": time.time() - t0})
            r = rows[-1]
            print(f"K={K:3d} trained {hops}-hop {mode:11s} H@1={r['hits1']:.3f} "
                  f"F1={r['f1']:.3f} Proof={r['proof_rate']:.1%} rec={r['obs_recall']:.3f} ovf={ovf}")

# ---------------------------------------------------------------- Supercondriaque, m=2
print("\nSupercondriaque duplicate-use case under multiplicity m:")
target = next(q for q in queries[3] if "Supercondriaque" in q.question_text)
base = TrainedPathRetriever(classifiers, top_k=K_MAIN_M2)


class MultiplicityWrapper:
    def __init__(self, base, m):
        self.base, self.m = base, m

    def propose_observations(self, instance):
        return Counter({a: c * self.m for a, c in self.base.propose_observations(instance).items()})


sup_result = {}
for m in (1, 2):
    res = run_sell_pipeline(target, MultiplicityWrapper(base, m), mode="sell", budget=BUDGET)
    got = "Dany Boon" in res["predicted"]
    sup_result[m] = got
    print(f"  m={m}: 'Dany Boon' proved = {got} (n_proved={res['n_proved']})")

# ---------------------------------------------------------------- save
pd.DataFrame(rows).to_csv(OUT / "exp_m2_trained.csv", index=False)
pd.DataFrame(perquery).to_csv(OUT / "exp_m2_perquery.csv", index=False)
summary = {"clf_test_accuracy": clf_acc, "verify_total": verify_total,
           "verify_pass": verify_pass, "budget_violations": budget_viol,
           "supercondriaque_proved_by_m": {str(k): v for k, v in sup_result.items()}}
(OUT / "exp_m2_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
print("DONE")
