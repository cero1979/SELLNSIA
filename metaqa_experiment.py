"""Experiment M1: SELL resource-aware KGQA on the real MetaQA benchmark.

Protocol mirrors Experiment R1 (Wikidata + real retriever) from SELLExpNSy.ipynb:
- KB: MetaQA kb.txt (134,741 triples, 9 relations).
- Questions: MetaQA vanilla 1/2/3-hop test sets (sampled, seed 42).
- Relation-path recovery: exhaustive search over (relation, direction) paths of
  length h from the bracketed topic entity; a question is admitted iff some path's
  terminal set (minus topic) equals the gold answer set exactly. This substitutes
  the official qtype annotations (absent from the mirror used).
- Resource-aware split: first-hop facts (and second-hop facts for 3-hop) are
  observation-required (linear, must be retrieved); deeper facts are persistent.
- Grounding: relevance-guided over the topic neighborhood (paper Sec. 5.2).
- Retrievers: TF-IDF, BM25, and a two-round iterative TF-IDF variant. Budget k=5.

Usage (from the repository root; sell_core.py must sit next to this script):
    PYTHONHASHSEED=0 python metaqa_experiment.py
MetaQA data (kb.txt + vanilla qa_test files, CC license) is auto-downloaded to
./metaqa_data/ from the Hugging Face mirror camazlucas/MetaQA on first run.
Outputs (CSV tables + summary JSON) are written to ./results/.
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from sell_core import (Atom, GroundRule, QAInstance, SELLProver,  # noqa: E402
                       BM25Retriever, TfidfRetriever, run_neural_only,
                       run_sell_pipeline, verify_trace)

SEED = 42
DATA = HERE / "metaqa_data"
OUT = HERE / "results"
N_PER_HOP = 150
MAX_RULES, MAX_ANSWERS, MAX_OBS = 4000, 50, 150
K_VALUES = [25, 50, 100]
BUDGET = 5
K_MAIN = 50  # operating point for per-query stats / verification

METAQA_MIRROR = "https://huggingface.co/datasets/camazlucas/MetaQA/resolve/main"
METAQA_FILES = {
    "kb.txt": "kb/kb.txt",
    "1-hop/qa_test.txt": "1-hop/vanilla/qa_test.txt",
    "2-hop/qa_test.txt": "2-hop/vanilla/qa_test.txt",
    "3-hop/qa_test.txt": "3-hop/vanilla/qa_test.txt",
}
for dst, src in METAQA_FILES.items():
    target = DATA / dst
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {src} -> {target} ...")
        urllib.request.urlretrieve(f"{METAQA_MIRROR}/{src}", target)

# ---------------------------------------------------------------- KB loading
print("Loading MetaQA KB ...")
objects = defaultdict(lambda: defaultdict(set))   # rel -> subj -> {obj}
subjects = defaultdict(lambda: defaultdict(set))  # rel -> obj  -> {subj}
all_triples = []
entities = set()
with open(DATA / "kb.txt") as fh:
    for line in fh:
        s, r, o = line.rstrip("\n").split("|")
        objects[r][s].add(o)
        subjects[r][o].add(s)
        all_triples.append((s, r, o))
        entities.add(s)
        entities.add(o)
RELATIONS = sorted(objects.keys())
lower2canon = {}
for e in sorted(entities):  # sorted: process-independent canonical pick
    lower2canon.setdefault(e.lower(), e)
print(f"  {len(all_triples)} triples, {len(entities)} entities, relations: {RELATIONS}")

# Global terminal-type sets: candidates for a final step (rel, dir)
subj_of = {r: set(objects[r].keys()) for r in RELATIONS}
obj_of = {r: set(subjects[r].keys()) for r in RELATIONS}


def step(node, rel, direction):
    return objects[rel].get(node, set()) if direction == "f" else subjects[rel].get(node, set())


def edge_atom(node, nxt, rel, direction):
    """Canonical KB orientation: Atom(rel, (movie, x))."""
    return Atom(rel, (node, nxt)) if direction == "f" else Atom(rel, (nxt, node))


# ------------------------------------------------------- path recovery
def recover_path(topic, gold, hops):
    """First (rel,dir) path whose terminal set == gold under MetaQA semantics:
    chains may not revisit the topic entity at intermediate positions, and the
    topic is excluded from the answer set."""
    frontier = [((), {topic})]
    for i in range(hops):
        nxt_frontier = []
        for path, nodes in frontier:
            for r in RELATIONS:
                for d in ("f", "b"):
                    reach = set()
                    for n in nodes:
                        reach |= step(n, r, d)
                    if i < hops - 1:
                        reach.discard(topic)  # no topic revisit mid-chain
                    if reach:
                        nxt_frontier.append((path + ((r, d),), reach))
        frontier = nxt_frontier
    for path, nodes in frontier:
        if nodes - {topic} == gold:
            return path
    return None


# ------------------------------------------------------- instance builder
def build_instance(qid, question, topic, gold, path):
    hops = len(path)
    obs_required, persistent = set(), set()
    rules, seen = [], set()
    answers = set()

    if hops == 1:
        (r1, d1) = path[0]
        n1 = step(topic, r1, d1) - {topic}
        for x in sorted(n1):
            obs_required.add(edge_atom(topic, x, r1, d1))
            answers.add(x)
        if d1 == "f":
            goal_pred = r1  # goal atom == canonical KB atom
        else:
            goal_pred = "ans1"
            for x in sorted(n1):  # budget-free orientation rules (Def. 4, n>=0)
                rid = f"o1_{x}"
                rules.append(GroundRule(head=Atom("ans1", (topic, x)),
                                        body=(edge_atom(topic, x, r1, d1),),
                                        budgeted=False, rule_id=rid))
        cand_pool = (obj_of if d1 == "f" else subj_of)[r1]
    elif hops == 2:
        (r1, d1), (r2, d2) = path
        goal_pred = "ans2"
        for e1 in sorted(step(topic, r1, d1)):
            a1 = edge_atom(topic, e1, r1, d1)
            obs_required.add(a1)
            for e2 in sorted(step(e1, r2, d2)):
                if e2 == topic:
                    continue
                a2 = edge_atom(e1, e2, r2, d2)
                persistent.add(a2)
                answers.add(e2)
                rid = f"c2_{e1}_{e2}"
                if rid not in seen:
                    rules.append(GroundRule(head=Atom("ans2", (topic, e2)),
                                            body=(a1, a2), budgeted=True, rule_id=rid))
                    seen.add(rid)
        cand_pool = (obj_of if d2 == "f" else subj_of)[r2]
    else:
        (r1, d1), (r2, d2), (r3, d3) = path
        goal_pred = "ans3"
        for e1 in sorted(step(topic, r1, d1)):
            a1 = edge_atom(topic, e1, r1, d1)
            for e2 in sorted(step(e1, r2, d2)):
                if e2 == topic:
                    continue  # MetaQA semantics: no topic revisit
                a2 = edge_atom(e1, e2, r2, d2)
                rid1 = f"i3_{e1}_{e2}"
                have_tail = False
                for e3 in sorted(step(e2, r3, d3)):
                    if e3 == topic:
                        continue
                    a3 = edge_atom(e2, e3, r3, d3)
                    rid2 = f"c3_{e2}_{e3}"
                    if rid2 not in seen:
                        rules.append(GroundRule(head=Atom("ans3", (topic, e3)),
                                                body=(Atom("int3", (topic, e2)), a3),
                                                budgeted=True, rule_id=rid2))
                        seen.add(rid2)
                    persistent.add(a3)
                    answers.add(e3)
                    have_tail = True
                if have_tail:
                    obs_required.add(a1)
                    obs_required.add(a2)
                    if rid1 not in seen:
                        rules.append(GroundRule(head=Atom("int3", (topic, e2)),
                                                body=(a1, a2), budgeted=True, rule_id=rid1))
                        seen.add(rid1)
                if len(rules) > MAX_RULES:
                    return None
        cand_pool = (obj_of if d3 == "f" else subj_of)[r3]

    if answers != gold:
        return None
    if len(rules) > MAX_RULES or len(answers) > MAX_ANSWERS or len(obs_required) > MAX_OBS:
        return None
    persistent -= obs_required  # retrieval-hop facts are never persistent
    candidates = (cand_pool & entities) - {topic}
    return QAInstance(
        question_id=qid, question_text=question.replace("[", "").replace("]", ""),
        hops=hops, topic_entity=topic, answer_entities=set(gold),
        candidate_universe=candidates, persistent_facts=persistent,
        obs_required_facts=obs_required, kg_facts=persistent | obs_required,
        ground_rules=rules, goal_predicate=goal_pred,
        goal_template=f"{goal_pred}({topic}, ?)")


def load_questions(hops):
    rng = random.Random(SEED + hops)
    lines = (DATA / f"{hops}-hop" / "qa_test.txt").read_text().rstrip("\n").split("\n")
    rng.shuffle(lines)
    stats = Counter()
    instances = []
    for line in lines:
        if len(instances) >= N_PER_HOP:
            break
        q, ans = line.split("\t")
        m = re.search(r"\[(.+?)\]", q)
        if not m:
            stats["no_topic"] += 1
            continue
        topic = lower2canon.get(m.group(1).lower())
        if topic is None:
            stats["topic_not_in_kb"] += 1
            continue
        gold = set()
        ok = True
        for a in ans.split("|"):
            c = lower2canon.get(a.lower(), a)
            if c not in entities:
                ok = False
                break
            gold.add(c)
        if not ok or not gold:
            stats["answer_not_in_kb"] += 1
            continue
        path = recover_path(topic, gold, hops)
        if path is None:
            stats["no_exact_path"] += 1
            continue
        inst = build_instance(f"mq{hops}_{len(instances)}", q, topic, gold, path)
        if inst is None:
            stats["capped"] += 1
            continue
        stats["admitted"] += 1
        instances.append(inst)
    return instances, stats


print("Building QA instances ...")
t0 = time.time()
queries = {}
build_stats = {}
for h in (1, 2, 3):
    queries[h], build_stats[h] = load_questions(h)
    print(f"  {h}-hop: {len(queries[h])} admitted  {dict(build_stats[h])}")
print(f"  instance building: {time.time()-t0:.1f}s")

# ------------------------------------------------------- retrieval corpus
print("Building retrieval corpus over full KB ...")
corpus_atoms = [Atom(r, (s, o)) for (s, r, o) in all_triples]
corpus_strings = [f"{s} {r.replace('_', ' ')} {o}" for (s, r, o) in all_triples]
tfidf = TfidfVectorizer(analyzer="word", lowercase=True,
                        token_pattern=r"(?u)\b\w+\b", sublinear_tf=True, norm="l2")
corpus_matrix = tfidf.fit_transform(corpus_strings)
bm25_index = BM25Okapi([s.lower().split() for s in corpus_strings])
print(f"  TF-IDF matrix {corpus_matrix.shape}; BM25 ready")

# ------------------------------------------------------- experiment loops
class IterativeTfidfRetriever:
    """Two-round retrieval (PullNet-style) under the same observation budget K:
    round 1 retrieves ceil(K/2) triples for the question; entity names from
    round-1 triples expand the query; round 2 retrieves the remaining budget.
    Retriever-agnostic observation proposal; SELL pipeline unchanged."""

    def __init__(self, tfidf, corpus_matrix, corpus_atoms, top_k=50):
        self.top_k = top_k
        self.k1 = (top_k + 1) // 2
        self.base = TfidfRetriever(tfidf, corpus_matrix, corpus_atoms, top_k=self.k1)

    def propose_observations(self, instance):
        round1 = self.base.retrieve_with_scores(instance.question_text, self.k1)
        ents = []
        for atom, _ in round1:
            ents.extend(atom.args)
        expanded = instance.question_text + " " + " ".join(dict.fromkeys(ents))
        round2 = self.base.retrieve_with_scores(expanded, self.top_k)
        obs = Counter()
        for atom, _ in round1:
            obs[atom] = 1
        for atom, _ in round2:
            if len(obs) >= self.top_k:
                break
            obs[atom] = 1
        return obs


class CachedRetriever:
    """Cache propose_observations per question (modes share proposals)."""

    def __init__(self, base):
        self.base = base
        self._cache = {}

    def propose_observations(self, instance):
        key = instance.question_id
        if key not in self._cache:
            self._cache[key] = self.base.propose_observations(instance)
        return Counter(self._cache[key])


MODES = ["sell", "classic_lp", "neural_only"]
rows, recall_rows, perquery_rows = [], [], []
verify_total = verify_pass = budget_viol = 0

for K in K_VALUES:
    retrievers = {
        "tfidf": CachedRetriever(TfidfRetriever(tfidf, corpus_matrix, corpus_atoms, top_k=K)),
        "bm25": CachedRetriever(BM25Retriever(bm25_index, corpus_atoms, top_k=K)),
        "tfidf_iter": CachedRetriever(IterativeTfidfRetriever(tfidf, corpus_matrix, corpus_atoms, top_k=K)),
    }
    for rname, retr in retrievers.items():
        for hops, qs in queries.items():
            recalls = []
            for q in qs:
                obs = retr.propose_observations(q)
                if q.obs_required_facts:
                    recalls.append(len(set(obs) & q.obs_required_facts) / len(q.obs_required_facts))
            recall_rows.append({"retriever": rname, "K": K, "hops": hops,
                                "obs_recall": float(np.mean(recalls))})
            for mode in MODES:
                t0 = time.time()
                h1l, f1l, prl, ovf, ntr = [], [], [], 0, 0
                for q in qs:
                    if mode == "neural_only":
                        res = run_neural_only(q, retr)
                    else:
                        res = run_sell_pipeline(q, retr, mode=mode, budget=BUDGET)
                    h1l.append(1.0 if res["hits_at_1"] else 0.0)
                    f1l.append(res["f1"])
                    prl.append(1.0 if res["proved"] else 0.0)
                    for tr in res.get("traces", []):
                        ntr += 1
                        if tr.budget_used > BUDGET:
                            ovf += 1
                    if mode == "sell" and rname in ("tfidf", "tfidf_iter") and K == K_MAIN:
                        perquery_rows.append({
                            "question_id": q.question_id, "hops": hops,
                            "retriever": rname,
                            "hits_at_1": res["hits_at_1"], "f1": res["f1"],
                            "proved": res["proved"], "n_proved": res["n_proved"]})
                        for tr in res["traces"]:
                            verify_total += 1
                            acc, _ = verify_trace(tr, q.persistent_facts,
                                                  Counter(retr.propose_observations(q)), BUDGET)
                            verify_pass += int(acc)
                            budget_viol += int(tr.budget_used > BUDGET)
                rows.append({"retriever": rname, "K": K, "hops": hops, "mode": mode,
                             "hits1": float(np.mean(h1l)), "f1": float(np.mean(f1l)),
                             "proof_rate": float(np.mean(prl)), "budget_overflow": ovf,
                             "total_traces": ntr, "n_queries": len(qs),
                             "time_s": time.time() - t0})
                r = rows[-1]
                print(f"K={K:3d} {rname:5s} {hops}-hop {mode:11s} "
                      f"H@1={r['hits1']:.3f} F1={r['f1']:.3f} "
                      f"Proof={r['proof_rate']:.1%} ovf={ovf} ({r['time_s']:.0f}s)")

# ------------------------------------------------------- budget sensitivity
budget_rows = []
retr50 = CachedRetriever(IterativeTfidfRetriever(tfidf, corpus_matrix, corpus_atoms, top_k=K_MAIN))
for k in (1, 2, 5):
    for hops in (2, 3):
        h1l, prl = [], []
        for q in queries[hops]:
            res = run_sell_pipeline(q, retr50, mode="sell", budget=k)
            h1l.append(1.0 if res["hits_at_1"] else 0.0)
            prl.append(1.0 if res["proved"] else 0.0)
        budget_rows.append({"budget_k": k, "hops": hops,
                            "hits1": float(np.mean(h1l)), "proof_rate": float(np.mean(prl))})
        print(f"budget k={k} {hops}-hop H@1={budget_rows[-1]['hits1']:.3f}")

# ------------------------------------------------------- save
OUT.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(OUT / "exp_m1_metaqa.csv", index=False)
pd.DataFrame(recall_rows).to_csv(OUT / "exp_m1_recall.csv", index=False)
pd.DataFrame(budget_rows).to_csv(OUT / "exp_m1_budget.csv", index=False)
pd.DataFrame(perquery_rows).to_csv(OUT / "exp_m1_perquery.csv", index=False)
summary = {"build_stats": {h: dict(s) for h, s in build_stats.items()},
           "verify_total": verify_total, "verify_pass": verify_pass,
           "budget_violations": budget_viol,
           "n_queries": {h: len(qs) for h, qs in queries.items()}}
(OUT / "exp_m1_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
print("DONE")
