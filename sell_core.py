"""SELL core extracted verbatim from SELLExpNSy.ipynb (SELLNSIA repo)."""
from __future__ import annotations
import random, time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

SEED = 42


@dataclass(frozen=True)
class Atom:
    """
    A ground atom, e.g., parent(alice, bob).
    Frozen so it can be used as dict key / set element.
    """
    predicate: str
    args: Tuple[str, ...]

    def __str__(self) -> str:
        if not self.args:
            return self.predicate
        return f"{self.predicate}({', '.join(self.args)})"

    def __repr__(self) -> str:
        return str(self)

# Special budget-token atom (Section 4.1, Definition 4)
BUDGET_ATOM = Atom("B", ())


@dataclass(frozen=True)
class GroundRule:
    """
    A ground rule (Definition 4):
        B ⊗ body[0] ⊗ ... ⊗ body[n-1]  ⊸  head
    If `budgeted` is True, one budget token is consumed per application.
    `rule_id` is used for trace readability.
    """
    head: Atom
    body: Tuple[Atom, ...]          # premises (atoms to prove)
    budgeted: bool = True           # whether this rule requires a budget token
    rule_id: str = ""               # human-readable identifier

    def __str__(self) -> str:
        parts = []
        if self.budgeted:
            parts.append("B")
        parts.extend(str(a) for a in self.body)
        body_str = " ⊗ ".join(parts) if parts else "⊤"
        return f"[{self.rule_id}] {body_str} ⊸ {self.head}"

    def __repr__(self) -> str:
        return str(self)


@dataclass
class ProofStep:
    """One rule-application step in a proof trace (Definition 8)."""
    rule: GroundRule
    consumed_observations: List[Atom]   # obs atoms consumed in this step
    budget_consumed: bool               # True if a budget token was consumed

    def __str__(self) -> str:
        obs_str = ", ".join(str(a) for a in self.consumed_observations)
        bud_str = " [−1 budget]" if self.budget_consumed else ""
        return f"Fire {self.rule.rule_id}: {self.rule.head}{bud_str} (obs used: {obs_str or 'none'})"


@dataclass
class ProofTrace:
    """
    Complete proof trace Tr(π) (Definition 8).
    Contains the sequence of grounded rule instances fired during proof search.
    """
    goal: Atom
    steps: List[ProofStep] = field(default_factory=list)
    success: bool = False
    budget_used: int = 0
    observations_consumed: int = 0

    @property
    def depth(self) -> int:
        """Number of rule applications (|Tr(π)|)."""
        return len(self.steps)

    def summary(self) -> str:
        status = "PROVED" if self.success else "FAILED"
        lines = [f"Goal: {self.goal}  [{status}]",
                 f"  Rule applications: {self.depth}",
                 f"  Budget consumed:   {self.budget_used}",
                 f"  Observations used: {self.observations_consumed}"]
        for i, s in enumerate(self.steps, 1):
            lines.append(f"  Step {i}: {s}")
        return "\n".join(lines)


print("Core data structures defined: Atom, GroundRule, ProofStep, ProofTrace")
# Quick sanity check
_a = Atom("parent", ("alice", "bob"))
_r = GroundRule(head=Atom("grandparent", ("alice", "carol")),
                body=(Atom("parent", ("alice", "bob")), Atom("parent", ("bob", "carol"))),
                budgeted=True, rule_id="gp_rule_1")
print(f"  Example atom:  {_a}")
print(f"  Example rule:  {_r}")



# ============================================================
# Cell 4: SELL Focused Proof-Search Engine
# ============================================================
# Implements Algorithm 1 (proof-search core) and the multiplicative
# rules + labelled dereliction discipline from Section 3.
#
# Design:
#   - psi_kg  (Set[Atom])     : persistent KG facts        (label kg, reusable)
#   - psi_rl  (List[GroundRule]): persistent ground rules   (label rl, reusable)
#   - gamma_obs (Counter[Atom]): consumable observations    (label obs, linear)
#   - budget  (int)            : remaining budget tokens    (label bud, linear)
#
# Proof search is backward-chaining: given a goal atom G, we try:
#   1. Ax from persistent facts (Use_u + Ax)
#   2. Ax from linear observations (Use_ℓ + Ax, consuming one copy)
#   3. ⊸L with each applicable rule whose head matches G:
#      - Prove each body atom recursively (multiplicative context splitting)
#      - Consume one budget token if the rule is budgeted
#
# The prover records every rule application in a ProofTrace object.

class SELLProver:
    """
    Focused proof-search engine for the grounded SELL-Horn fragment.
    
    Parameters
    ----------
    psi_kg : set of Atom
        Persistent KG facts (reusable, label kg ∈ U).
    psi_rl : list of GroundRule
        Persistent ground rules (reusable, label rl ∈ U).
    mode : str
        Resource-management mode:
        - "sell"           : full SELL (obs consumable, budget enforced)
        - "no_budget"      : obs consumable, but no budget limit
        - "obs_persistent" : obs treated as persistent (reusable), budget enforced
        - "classic_lp"     : all facts persistent, no budget (classical LP)
    max_depth : int
        Hard recursion limit to prevent infinite loops (default 50).
    """

    def __init__(self, psi_kg: Set[Atom], psi_rl: List[GroundRule],
                 mode: str = "sell", max_depth: int = 50):
        self.psi_kg = set(psi_kg)
        self.psi_rl = list(psi_rl)
        self.mode = mode
        self.max_depth = max_depth

        # Build a head-index for fast rule lookup
        self._rules_by_head: Dict[Atom, List[GroundRule]] = defaultdict(list)
        for r in self.psi_rl:
            self._rules_by_head[r.head].append(r)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def prove(self, goal: Atom, gamma_obs: Counter, budget: int
              ) -> Tuple[bool, ProofTrace, Counter]:
        """
        Attempt to prove `goal` from the given linear context.
        Returns (success, trace, remaining_observations).
        
        The third element is the observation counter AFTER consumption:
        useful for shared-context proof search across multiple goals
        (the SELL resource discipline).
        """
        ok, trace_steps, remaining_obs, _, obs_used, bud_used = self._prove(
            goal, Counter(gamma_obs), budget, depth=0
        )
        trace = ProofTrace(
            goal=goal, steps=trace_steps, success=ok,
            budget_used=bud_used, observations_consumed=obs_used,
        )
        return ok, trace, remaining_obs

    # ------------------------------------------------------------------
    # Internal recursive search
    # ------------------------------------------------------------------
    def _prove(self, goal: Atom, obs: Counter, budget: int, depth: int
               ) -> Tuple[bool, List[ProofStep], Counter, int, int, int]:
        """
        Returns: (success, steps, remaining_obs, remaining_budget,
                  total_obs_consumed, total_budget_consumed)
        """
        if depth > self.max_depth:
            return False, [], obs, budget, 0, 0

        # ── Base case 1: Ax from persistent KG facts (Use_u + Ax) ────
        if goal in self.psi_kg:
            return True, [], obs, budget, 0, 0

        # ── Base case 2: Ax from consumable observations (Use_ℓ + Ax) ─
        if self.mode in ("sell", "no_budget"):
            # Linear consumption: obs is consumed
            if obs.get(goal, 0) > 0:
                new_obs = Counter(obs)
                new_obs[goal] -= 1
                if new_obs[goal] <= 0:
                    del new_obs[goal]
                return True, [], new_obs, budget, 1, 0
        elif self.mode == "obs_persistent":
            # Observations treated as persistent (ablation)
            if obs.get(goal, 0) > 0:
                return True, [], obs, budget, 0, 0
        elif self.mode == "classic_lp":
            # All observations treated as persistent, no budget
            if obs.get(goal, 0) > 0:
                return True, [], obs, budget, 0, 0

        # ── Try rules via ⊸L ─────────────────────────────────────────
        candidate_rules = self._rules_by_head.get(goal, [])
        for rule in candidate_rules:
            # Budget check
            effective_budgeted = rule.budgeted and self.mode not in ("no_budget", "classic_lp")
            if effective_budgeted and budget <= 0:
                continue  # not enough budget

            # Attempt to prove all body atoms (multiplicative context splitting)
            current_obs = Counter(obs)
            current_budget = budget - (1 if effective_budgeted else 0)
            all_proved = True
            sub_steps: List[ProofStep] = []
            total_obs = 0
            total_bud = 1 if effective_budgeted else 0
            consumed_obs_this_rule: List[Atom] = []

            for body_atom in rule.body:
                ok, steps, current_obs, current_budget, o_used, b_used = self._prove(
                    body_atom, current_obs, current_budget, depth + 1
                )
                if not ok:
                    all_proved = False
                    break
                sub_steps.extend(steps)
                total_obs += o_used
                total_bud += b_used

            if all_proved:
                # Record this rule application as a ProofStep
                step = ProofStep(
                    rule=rule,
                    consumed_observations=consumed_obs_this_rule,
                    budget_consumed=effective_budgeted,
                )
                all_steps = sub_steps + [step]
                return True, all_steps, current_obs, current_budget, total_obs, total_bud

        # No rule succeeded
        return False, [], obs, budget, 0, 0


print("SELLProver class defined.")
print("Modes available: 'sell', 'no_budget', 'obs_persistent', 'classic_lp'")



# ============================================================
# Cell 5: Trace Verification — Algorithm 2
# ============================================================
# Independent verifier that replays a proof trace as multiset
# rewriting and checks resource correctness.
# Corresponds to Algorithm 2 (Trace verification) in the paper.

def verify_trace(trace: ProofTrace,
                 psi_kg: Set[Atom],
                 gamma_obs: Counter,
                 budget: int) -> Tuple[bool, str]:
    """
    Verify a proof trace via multiset-rewriting replay (Algorithm 2).
    
    Parameters
    ----------
    trace : ProofTrace
        The trace produced by the SELL prover.
    psi_kg : set of Atom
        Persistent KG facts (always available, not consumed).
    gamma_obs : Counter of Atom
        Initial consumable observations.
    budget : int
        Initial budget (k).
    
    Returns
    -------
    (accepted, message) : (bool, str)
        Whether the trace is valid and a diagnostic message.
    """
    if not trace.success:
        return False, "Trace reports failure; nothing to verify."

    # Initialize state S = F_obs ⊎ {bud, ..., bud}  (Definition 10)
    state = Counter(gamma_obs)
    remaining_budget = budget

    # Extract rule firings from the trace
    rule_firings = [step for step in trace.steps]

    for i, step in enumerate(rule_firings):
        rule = step.rule

        # Check budget
        if step.budget_consumed:
            if remaining_budget <= 0:
                return False, f"REJECT at step {i+1}: budget exhausted before firing {rule.rule_id}."
            remaining_budget -= 1

        # Check and consume body atoms
        for body_atom in rule.body:
            if body_atom in psi_kg:
                continue  # persistent premise, always available
            elif state.get(body_atom, 0) > 0:
                state[body_atom] -= 1
                if state[body_atom] <= 0:
                    del state[body_atom]
            else:
                return False, (f"REJECT at step {i+1}: premise {body_atom} of rule "
                               f"{rule.rule_id} not available in state.")

        # Add the head to the state
        state[rule.head] += 1

    # Check if the goal is available
    goal = trace.goal
    if goal in psi_kg or state.get(goal, 0) > 0:
        return True, f"ACCEPT: goal {goal} is available after {len(rule_firings)} firings."
    else:
        return False, f"REJECT: goal {goal} not in final state."


# ── Quick test with the running example ──────────────────────
print("Trace verifier defined.")
print()

# Running example (Example 1 / Section 4.4):
# KG fact: parent(alice, bob)  [persistent]
# Obs:     parent(bob, carol)  [consumable]
# Rule:    B ⊗ parent(alice,bob) ⊗ parent(bob,carol) ⊸ grandparent(alice,carol)
# Budget k=1

p1 = Atom("parent", ("alice", "bob"))
p2 = Atom("parent", ("bob", "carol"))
gp = Atom("grandparent", ("alice", "carol"))
rule_gp = GroundRule(head=gp, body=(p1, p2), budgeted=True, rule_id="gp_abc")

prover = SELLProver(psi_kg={p1}, psi_rl=[rule_gp], mode="sell")
ok, trace, _ = prover.prove(goal=gp, gamma_obs=Counter({p2: 1}), budget=1)

print("─── Running Example (Section 4.4) ───")
print(trace.summary())
print()

# Verify the trace independently
accepted, msg = verify_trace(trace, psi_kg={p1}, gamma_obs=Counter({p2: 1}), budget=1)
print(f"Trace verification: {msg}")

# Demonstrate that with budget=0, the proof fails
ok0, trace0, _ = prover.prove(goal=gp, gamma_obs=Counter({p2: 1}), budget=0)
print(f"\nWith budget=0: {'PROVED' if ok0 else 'FAILED (expected — no budget)'}")


@dataclass
class QAInstance:
    """
    A single QA instance for the benchmark.
    
    Key design: facts are split into two groups:
      - persistent_facts: always available in Ψ (trusted KG backbone)
      - obs_required_facts: ONLY available through observations (Γ)
    
    Under noise (η > 0), obs_required_facts may be corrupted,
    and the neural scorer may inject false triples that could
    participate in rule chains (adversarial noise).
    """
    question_id: str
    question_text: str
    hops: int                           # 1, 2, or 3
    topic_entity: str                   # anchor entity
    answer_entities: Set[str]           # ground-truth answers
    candidate_universe: Set[str]        # C (finite candidate set)
    persistent_facts: Set[Atom]         # F_kg (persistent, always in Ψ)
    obs_required_facts: Set[Atom]       # facts available ONLY via observations (Γ)
    kg_facts: Set[Atom]                 # all facts = persistent ∪ obs_required
    ground_rules: List[GroundRule]      # R_gr (grounded rules)
    goal_predicate: str                 # predicate for goal atoms
    goal_template: str                  # e.g. "actor_of_director(X, dir)"




# ============================================================
# Cell 9: End-to-End Pipeline (Algorithm 1) + All Baselines
# ============================================================
# Implements the full resource-aware NeSy KG/KB-QA pipeline
# and all baseline/ablation configurations.

def _split_kg_obs(instance: QAInstance, obs: Counter,
                  persistent_obs: bool = False
                  ) -> Tuple[Set[Atom], Counter]:
    """
    Decide which facts go into Ψ (persistent) and which into Γ (linear).
    
    - instance.persistent_facts → always in Ψ
    - Neural observations (obs) → in Γ (consumable) UNLESS persistent_obs=True
    
    For 'obs_persistent' / 'classic_lp' ablations: everything goes to Ψ.
    """
    psi_kg = set(instance.persistent_facts)
    if persistent_obs:
        # Ablation: treat observations as persistent too
        psi_kg = psi_kg | set(obs.keys())
        return psi_kg, Counter()
    else:
        return psi_kg, Counter(obs)


def run_sell_pipeline(instance: QAInstance,
                      scorer: NeuralScorer,
                      mode: str = "sell",
                      budget: int = 5,
                      ) -> Dict[str, Any]:
    """
    Full NeSy KG/KB-QA pipeline (Algorithm 1).
    
    Parameters
    ----------
    instance : QAInstance
    scorer : NeuralScorer
    mode : str
        One of: 'sell', 'no_budget', 'obs_persistent', 'classic_lp'
    budget : int
        Number of budget tokens (k).
    
    Returns
    -------
    dict with keys: predicted, correct, proved, trace, time_ms, budget_used, obs_consumed
    """
    t0 = time.perf_counter()

    # Step 1-2: Neural proposer generates observations
    obs = scorer.propose_observations(instance)

    # Step 3: Build contexts
    persistent_obs = (mode in ("obs_persistent", "classic_lp"))
    psi_kg, gamma_obs = _split_kg_obs(instance, obs, persistent_obs=persistent_obs)

    # Step 4: Build prover
    effective_budget = budget if mode not in ("no_budget", "classic_lp") else 999
    prover = SELLProver(psi_kg=psi_kg, psi_rl=instance.ground_rules,
                        mode=mode, max_depth=30)

    # Step 5: For entity-answering queries, iterate over candidates
    #
    # Each candidate is proved independently with a fresh observation context.
    # The SELL resource discipline operates WITHIN each proof tree:
    # consumable observations are used at most once per proof, and budget
    # limits the number of budgeted rule applications per proof.
    proved_answers = []
    traces = []

    for candidate in sorted(instance.candidate_universe):
        goal = Atom(instance.goal_predicate, (instance.topic_entity, candidate))
        ok, trace, _ = prover.prove(goal, gamma_obs=Counter(gamma_obs),
                                    budget=effective_budget)
        if ok:
            proved_answers.append(candidate)
            traces.append(trace)

    elapsed = (time.perf_counter() - t0) * 1000  # ms

    # Determine prediction
    predicted = set(proved_answers)
    correct = predicted == instance.answer_entities

    # For Hits@1: check if any correct answer was proved
    hits_at_1 = len(predicted & instance.answer_entities) > 0

    # Compute precision, recall, F1
    tp = len(predicted & instance.answer_entities)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(instance.answer_entities) if instance.answer_entities else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "question_id": instance.question_id,
        "hops": instance.hops,
        "predicted": predicted,
        "gold": instance.answer_entities,
        "hits_at_1": hits_at_1,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": correct,
        "proved": len(proved_answers) > 0,
        "n_proved": len(proved_answers),
        "traces": traces,
        "time_ms": elapsed,
        "budget_used": sum(t.budget_used for t in traces),
        "obs_consumed": sum(t.observations_consumed for t in traces),
    }


def run_neural_only(instance: QAInstance,
                    scorer: NeuralScorer) -> Dict[str, Any]:
    """
    Neural-only baseline: predicts the most common candidate entity
    appearing in the neural observations, without symbolic reasoning.
    """
    t0 = time.perf_counter()

    obs = scorer.propose_observations(instance)

    # Count entity mentions in observations as candidates
    entity_scores: Counter = Counter()
    for atom, count in obs.items():
        for arg in atom.args:
            if arg in instance.candidate_universe:
                entity_scores[arg] += count

    # Predict top-scoring candidates
    if entity_scores:
        max_score = entity_scores.most_common(1)[0][1]
        predicted = {e for e, s in entity_scores.items() if s == max_score}
    else:
        predicted = set()

    elapsed = (time.perf_counter() - t0) * 1000

    tp = len(predicted & instance.answer_entities)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(instance.answer_entities) if instance.answer_entities else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "question_id": instance.question_id,
        "hops": instance.hops,
        "predicted": predicted,
        "gold": instance.answer_entities,
        "hits_at_1": len(predicted & instance.answer_entities) > 0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": predicted == instance.answer_entities,
        "proved": False,
        "n_proved": 0,
        "traces": [],
        "time_ms": elapsed,
        "budget_used": 0,
        "obs_consumed": 0,
    }


print("Pipeline functions defined:")
print("  - run_sell_pipeline(instance, scorer, mode, budget)")
print("  - run_neural_only(instance, scorer)")
print("Modes: 'sell', 'no_budget', 'obs_persistent', 'classic_lp'")


class TfidfRetriever:
    """
    Real TF-IDF retriever over KG triples.
    
    Replaces the simulated NeuralScorer with a genuine
    retrieval component. No training required.
    
    Compatible with the NeuralScorer interface:
      retriever.propose_observations(instance) → Counter[Atom]
    """
    
    def __init__(self, tfidf_vectorizer, corpus_matrix, corpus_atoms, top_k=25):
        self.tfidf = tfidf_vectorizer
        self.corpus_matrix = corpus_matrix
        self.corpus_atoms = corpus_atoms
        self.top_k = top_k
    
    def propose_observations(self, instance) -> Counter:
        """
        Retrieve Top-K triples most similar to the question text.
        Returns a Counter[Atom] compatible with the pipeline.
        """
        question = instance.question_text
        q_vec = self.tfidf.transform([question])
        sims = cosine_similarity(q_vec, self.corpus_matrix).flatten()
        
        # Get top-K indices
        top_indices = np.argsort(sims)[::-1][:self.top_k]
        
        retrieved = Counter()
        for idx in top_indices:
            if sims[idx] > 0:  # only non-zero similarity
                retrieved[self.corpus_atoms[idx]] = 1
        
        return retrieved
    
    def retrieve_with_scores(self, question: str, top_k: int = None):
        """Return (atom, score) pairs for analysis."""
        k = top_k or self.top_k
        q_vec = self.tfidf.transform([question])
        sims = cosine_similarity(q_vec, self.corpus_matrix).flatten()
        top_indices = np.argsort(sims)[::-1][:k]
        return [(self.corpus_atoms[i], float(sims[i])) for i in top_indices if sims[i] > 0]


class BM25Retriever:
    """
    BM25 retriever over KG triples.
    Standard probabilistic retrieval baseline (Robertson & Zaragoza, 2009).
    Compatible with NeuralScorer interface.
    """
    def __init__(self, bm25_index, corpus_atoms, top_k=25):
        self.bm25 = bm25_index
        self.corpus_atoms = corpus_atoms
        self.top_k = top_k

    def propose_observations(self, instance) -> Counter:
        question = instance.question_text.lower().split()
        scores = self.bm25.get_scores(question)
        top_indices = np.argsort(scores)[::-1][:self.top_k]
        retrieved = Counter()
        for idx in top_indices:
            if scores[idx] > 0:
                retrieved[self.corpus_atoms[idx]] = 1
        return retrieved

    def retrieve_with_scores(self, question: str, top_k=None):
        k = top_k or self.top_k
        tokens = question.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        return [(self.corpus_atoms[i], float(scores[i]))
                for i in top_indices if scores[i] > 0]