"""Experiment M3: observation-multiplicity sweep (m in {1,2,3}) for the
forward-chaining false-positive amplification setting (Experiment 5).

Reuses, verbatim, the Exp-5 machinery from SELLExpNSy.ipynb: scaled MovieKG,
comprehensive 3-hop instances, adversarial NeuralScorer, and the saturation
engines. Multiplicity m injects m independent linear copies of each retrieved
observation into the linear context. Classic LP is the m = infinity endpoint.

Usage (repo root, needs sell_core.py): PYTHONHASHSEED=0 python multiplicity_experiment.py
Outputs: results/exp_m3_multiplicity.csv (+ sanity check vs. exp5 CSV).
"""
from __future__ import annotations
import json, random, sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sell_core import Atom, GroundRule, QAInstance  # noqa: E402

SEED = 42


# ============================================================
# Cell 6: Movie-Domain Knowledge Graph Construction
# ============================================================
# Generates a synthetic movie KG with configurable size.
# Relations: acted_in, directed_by, has_genre, written_by
# Multi-hop rule schemas ground over the entity set.

class MovieKG:
    """
    Synthetic movie-domain knowledge graph (MetaQA-style).
    
    Parameters
    ----------
    n_movies : int      Number of movies (default 200).
    n_actors : int      Number of actors (default 120).
    n_directors : int   Number of directors (default 40).
    n_genres : int      Number of genres (default 15).
    n_writers : int     Number of writers (default 30).
    density : float     Average edges per movie (default 3.0).
    seed : int          Random seed.
    """

    RELATIONS = ["acted_in", "directed_by", "has_genre", "written_by"]

    def __init__(self, n_movies=200, n_actors=120, n_directors=40,
                 n_genres=15, n_writers=30, density=3.0, seed=42):
        rng = random.Random(seed)

        # ── Create entity pools ──────────────────────────────────
        self.movies   = [f"movie_{i}" for i in range(n_movies)]
        self.actors   = [f"actor_{i}" for i in range(n_actors)]
        self.directors = [f"dir_{i}" for i in range(n_directors)]
        self.genres   = [f"genre_{i}" for i in range(n_genres)]
        self.writers  = [f"writer_{i}" for i in range(n_writers)]

        self.all_entities = (self.movies + self.actors + self.directors +
                            self.genres + self.writers)

        # ── Generate triples ─────────────────────────────────────
        self.triples: List[Atom] = []
        self._acted_in: Dict[str, List[str]] = defaultdict(list)   # actor → [movie]
        self._directed_by: Dict[str, str] = {}                     # movie → director
        self._has_genre: Dict[str, List[str]] = defaultdict(list)  # movie → [genre]
        self._written_by: Dict[str, str] = {}                      # movie → writer

        for m in self.movies:
            # Assign 1-4 actors per movie
            n_act = rng.randint(1, min(4, n_actors))
            actors_m = rng.sample(self.actors, n_act)
            for a in actors_m:
                t = Atom("acted_in", (a, m))
                self.triples.append(t)
                self._acted_in[a].append(m)

            # Assign 1 director
            d = rng.choice(self.directors)
            self.triples.append(Atom("directed_by", (m, d)))
            self._directed_by[m] = d

            # Assign 1-3 genres
            n_gen = rng.randint(1, min(3, n_genres))
            genres_m = rng.sample(self.genres, n_gen)
            for g in genres_m:
                self.triples.append(Atom("has_genre", (m, g)))
                self._has_genre[m].append(g)

            # Assign 1 writer (70 % of movies)
            if rng.random() < 0.7:
                w = rng.choice(self.writers)
                self.triples.append(Atom("written_by", (m, w)))
                self._written_by[m] = w

        self.triple_set: Set[Atom] = set(self.triples)

        # ── Adjacency indices for query generation ───────────────
        self._build_indices()

    def _build_indices(self):
        """Build reverse indices for fast query generation."""
        self._movies_of_actor: Dict[str, List[str]] = defaultdict(list)
        self._actors_of_movie: Dict[str, List[str]] = defaultdict(list)
        self._director_of_movie: Dict[str, str] = {}
        self._movies_of_director: Dict[str, List[str]] = defaultdict(list)

        for t in self.triples:
            if t.predicate == "acted_in":
                self._movies_of_actor[t.args[0]].append(t.args[1])
                self._actors_of_movie[t.args[1]].append(t.args[0])
            elif t.predicate == "directed_by":
                self._director_of_movie[t.args[1]] = t.args[0]
                self._movies_of_director[t.args[1]].append(t.args[0])

    def stats(self) -> str:
        return (f"MovieKG: {len(self.movies)} movies, {len(self.actors)} actors, "
                f"{len(self.directors)} directors, {len(self.genres)} genres, "
                f"{len(self.writers)} writers, {len(self.triples)} triples")


# ── Instantiate the KG ──────────────────────────────────────
kg = MovieKG(n_movies=200, n_actors=120, n_directors=40,
             n_genres=15, n_writers=30, seed=SEED)
print(kg.stats())
print(f"Sample triples: {kg.triples[:5]}")


# ============================================================
# Cell 8: Neural Scorer & Observation Generator
# ============================================================
# Simulates the neural proposer/retriever (Section 5.1).
# Produces F_obs as a multiset of consumable observations
# with configurable noise.

class NeuralScorer:
    """
    Simulated neural scorer / retriever.
    
    For each QA instance, returns a multiset of candidate observations.
    
    Key design (adversarial noise):
    - obs_required_facts are the true facts that MUST come through observations.
    - With probability noise_rate, a true obs fact is replaced by a PLAUSIBLE
      false triple (same relation, real entities, but wrong combination).
    - This adversarial noise can participate in rule chains and generate
      false positive answers — exactly the scenario where SELL's linear
      consumption provides robustness.
    
    Parameters
    ----------
    kg : MovieKG
        The underlying knowledge graph.
    noise_rate : float
        Fraction of obs_required facts that are corrupted (0.0–1.0).
    top_k : int
        Maximum number of observations to propose.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, kg: MovieKG, noise_rate: float = 0.0,
                 top_k: int = 20, seed: int = 42):
        self.kg = kg
        self.noise_rate = noise_rate
        self.top_k = top_k
        self.rng = random.Random(seed)
        self._all_atoms = list(kg.triple_set)

    def _adversarial_false_triple(self, original: Atom) -> Atom:
        """
        Generate a plausible false triple with the SAME relation and
        entity type pattern as the original, but a wrong combination.
        This makes the noise adversarial: false triples can match rule bodies.
        """
        rel = original.predicate
        for _ in range(50):
            if rel == "acted_in":
                s = original.args[0]  # keep the actor, change the movie
                o = self.rng.choice(self.kg.movies)
            elif rel == "directed_by":
                s = self.rng.choice(self.kg.movies)  # change the movie 
                o = original.args[1]  # keep same director type entity
            elif rel == "has_genre":
                s = self.rng.choice(self.kg.movies)
                o = original.args[1]  # keep genre
            else:
                s = self.rng.choice(self.kg.movies)
                o = self.rng.choice(self.kg.writers)
            candidate = Atom(rel, (s, o))
            if candidate not in self.kg.triple_set and candidate != original:
                return candidate
        # Fallback: random false triple
        return self._random_false_triple()

    def _random_false_triple(self) -> Atom:
        """Generate a random triple that is NOT in the KG."""
        while True:
            rel = self.rng.choice(MovieKG.RELATIONS)
            if rel == "acted_in":
                s = self.rng.choice(self.kg.actors)
                o = self.rng.choice(self.kg.movies)
            elif rel == "directed_by":
                s = self.rng.choice(self.kg.movies)
                o = self.rng.choice(self.kg.directors)
            elif rel == "has_genre":
                s = self.rng.choice(self.kg.movies)
                o = self.rng.choice(self.kg.genres)
            else:  # written_by
                s = self.rng.choice(self.kg.movies)
                o = self.rng.choice(self.kg.writers)
            candidate = Atom(rel, (s, o))
            if candidate not in self.kg.triple_set:
                return candidate

    def propose_observations(self, instance: QAInstance) -> Counter:
        """
        Propose F_obs for a QA instance.
        
        Strategy:
          1. Start with obs_required_facts (the facts that MUST come via obs).
          2. With probability noise_rate, replace each with an adversarial false triple.
          3. Add distractor triples (random false) up to top_k.
          
        Classic LP (obs persistent) can reuse adversarial triples freely,
        generating more false positive answers. SELL's linear consumption
        limits each noisy triple to a single use.
        """
        obs_facts = list(instance.obs_required_facts)
        observations = []

        for fact in obs_facts:
            if self.rng.random() < self.noise_rate:
                # Replace with adversarial noise
                observations.append(self._adversarial_false_triple(fact))
            else:
                observations.append(fact)

        # Add random distractors
        n_distractors = min(5, self.top_k - len(observations))
        for _ in range(max(0, n_distractors)):
            observations.append(self._random_false_triple())

        return Counter(observations[:self.top_k])


# ============================================================
# Cell 14b: Forward-Chaining Saturation Engine
# ============================================================
# Implements the operational semantics (Definition 7, multiset
# rewriting) for computing ALL derivable facts from a given state.
#
# Two modes:
#   - Classic LP: observations are persistent (always available),
#     full fixed-point saturation.
#   - SELL: observations are consumable (each fires at most one
#     rule), budget-bounded.
#
# This is the ground-truth model to which proof search is
# sound (Theorem 1).

def saturate_classic_lp(persistent: Set[Atom],
                        observations: Set[Atom],
                        rules: List[GroundRule],
                        goal_predicate: str = ""
                        ) -> Set[Atom]:
    """
    Compute the full fixed-point saturation under Classic LP.
    
    ALL observations are treated as persistent: they can fire
    every matching rule body.  No budget limit.
    
    Returns the set of ALL derived atoms (goals reachable from
    the input observations + persistent KG via rule chains).
    """
    # Everything available (persistent + observations)
    available = set(persistent) | set(observations)
    derived: Set[Atom] = set()
    changed = True
    
    while changed:
        changed = False
        for rule in rules:
            if rule.head in derived or rule.head in available:
                continue
            # Check if ALL body atoms are available
            if all(a in available or a in derived for a in rule.body):
                derived.add(rule.head)
                available.add(rule.head)  # derived facts become available
                changed = True
    
    return derived


def saturate_sell(persistent: Set[Atom],
                  observations: Counter,
                  rules: List[GroundRule],
                  budget: int = 10,
                  seed: int = 42,
                  goal_predicate: str = ""
                  ) -> Tuple[Set[Atom], int]:
    """
    Compute derivable facts under SELL resource discipline.
    
    Observations are consumable: each occurrence fires at most
    one rule.  Derived intermediates are also consumable.
    Budget limits the number of budgeted rule firings.
    
    Uses a randomized greedy strategy over multiple orderings
    and returns the UNION of derivable facts (upper bound).
    
    Returns (derived_set, budget_used).
    """
    rng = random.Random(seed)
    all_derived: Set[Atom] = set()
    total_budget = 0
    
    # Run several random orderings to approximate maximal derivable set
    for trial in range(10):
        state = Counter(observations)
        remaining_budget = budget
        shuffled_rules = list(rules)
        rng.shuffle(shuffled_rules)
        trial_derived: Set[Atom] = set()
        trial_budget = 0
        
        changed = True
        max_iters = 500  # safety bound
        iters = 0
        while changed and remaining_budget > 0 and iters < max_iters:
            changed = False
            iters += 1
            for rule in shuffled_rules:
                # Skip non-budgeted overhead and check budget
                if rule.budgeted and remaining_budget <= 0:
                    continue
                
                # Check body: persistent atoms are free;
                # linear atoms must be in state
                body_ok = True
                to_consume: Counter = Counter()
                for a in rule.body:
                    if a in persistent:
                        continue  # always available
                    elif state[a] - to_consume[a] > 0:
                        to_consume[a] += 1
                    else:
                        body_ok = False
                        break
                
                if body_ok:
                    # Fire: consume body atoms, produce head
                    state -= to_consume
                    # clean zeros
                    state = +state
                    state[rule.head] += 1
                    trial_derived.add(rule.head)
                    if rule.budgeted:
                        remaining_budget -= 1
                        trial_budget += 1
                    changed = True
        
        all_derived |= trial_derived
        total_budget = max(total_budget, trial_budget)
    
    return all_derived, total_budget



# ============================================================
# Cell 14c: Scaled KG + Comprehensive 3-hop instances (Exp 5)
# ============================================================
# To demonstrate false-positive amplification, we need a KG where
# the gold answer set is a proper subset of all candidate entities.
# With the original 40 directors + comprehensive genre chains,
# nearly all directors are reachable (gold ~ 39/40), leaving no
# room for false positives.
#
# Solution: larger director pool (100) so that genre chains reach
# 40-70% of candidates, creating space for measurable FP differences.

# -- Scaled KG for false-positive analysis --------------------
kg_exp5 = MovieKG(n_movies=300, n_actors=150, n_directors=100,
                  n_genres=20, n_writers=50, seed=SEED + 100)
print(f"Scaled KG (Experiment 5): {kg_exp5.stats()}")

def generate_3hop_comprehensive(kg_src: MovieKG, n: int = 80,
                                 seed: int = 42) -> List[QAInstance]:
    """
    Generate 3-hop instances with COMPREHENSIVE ground rules
    and FULL answer sets computed via Classic LP saturation.
    
    Chain:
      acted_in(A, M1) + has_genre(M1, G) -> actor_genre(A, G)
      actor_genre(A, G) + has_genre(M2, G) + directed_by(M2, D) -> actor_genre_director(A, D)
    """
    rng = random.Random(seed + 10)
    instances = []

    # ALL persistent backbone
    all_persistent: Set[Atom] = set()
    for m in kg_src.movies:
        if m in kg_src._directed_by:
            all_persistent.add(Atom("directed_by", (m, kg_src._directed_by[m])))
        for g in kg_src._has_genre.get(m, []):
            all_persistent.add(Atom("has_genre", (m, g)))

    # Genre -> movies-with-director index
    genre_movies: Dict[str, List[str]] = defaultdict(list)
    for m in kg_src.movies:
        if m in kg_src._directed_by:
            for g in kg_src._has_genre.get(m, []):
                genre_movies[g].append(m)

    valid_actors = [a for a in kg_src.actors
                    if len(kg_src._movies_of_actor.get(a, [])) >= 2]
    rng.shuffle(valid_actors)

    for i, actor in enumerate(valid_actors[:n * 2]):
        movies_a = kg_src._movies_of_actor[actor]
        obs_required: Set[Atom] = set()
        for m in movies_a:
            obs_required.add(Atom("acted_in", (actor, m)))

        # Build COMPREHENSIVE rules
        rules: List[GroundRule] = []
        seen_rules: set = set()

        # Step-1: for ALL movies x genres in KG
        for m in kg_src.movies:
            for g in kg_src._has_genre.get(m, []):
                rid1 = f"ag_{actor}_{m}_{g}"
                if rid1 not in seen_rules:
                    rules.append(GroundRule(
                        head=Atom("actor_genre", (actor, g)),
                        body=(Atom("acted_in", (actor, m)),
                              Atom("has_genre", (m, g))),
                        budgeted=True, rule_id=rid1,
                    ))
                    seen_rules.add(rid1)

        # Step-2: for ALL (genre, movie2, director)
        for g in kg_src.genres:
            for m2 in genre_movies.get(g, []):
                d = kg_src._directed_by[m2]
                rid2 = f"agd_{actor}_{g}_{m2}_{d}"
                if rid2 not in seen_rules:
                    rules.append(GroundRule(
                        head=Atom("actor_genre_director", (actor, d)),
                        body=(Atom("actor_genre", (actor, g)),
                              Atom("has_genre", (m2, g)),
                              Atom("directed_by", (m2, d))),
                        budgeted=True, rule_id=rid2,
                    ))
                    seen_rules.add(rid2)

        # Compute TRUE answer set via Classic LP saturation
        true_derived = saturate_classic_lp(
            all_persistent, obs_required, rules,
            goal_predicate="actor_genre_director"
        )
        answers = {atom.args[1] for atom in true_derived
                   if atom.predicate == "actor_genre_director"}

        # FILTER: gold set <= 80% of directors (space for FPs)
        n_dirs = len(kg_src.directors)
        if not answers or len(answers) < 3 or len(answers) > int(0.80 * n_dirs):
            continue

        candidates = set(kg_src.directors)
        inst = QAInstance(
            question_id=f"3hop_comp_{i}",
            question_text=f"Who directed a movie in a genre of a movie {actor} acted in?",
            hops=3, topic_entity=actor,
            answer_entities=answers,
            candidate_universe=candidates,
            persistent_facts=all_persistent,
            obs_required_facts=obs_required,
            kg_facts=all_persistent | obs_required,
            ground_rules=rules,
            goal_predicate="actor_genre_director",
            goal_template=f"actor_genre_director({actor}, ?)",
        )
        instances.append(inst)
        if len(instances) >= n:
            break

    return instances


queries_3hop_comp = generate_3hop_comprehensive(kg_exp5, n=80, seed=SEED)
print(f"\nComprehensive 3-hop queries: {len(queries_3hop_comp)}")
if queries_3hop_comp:
    gold_sizes = [len(q.answer_entities) for q in queries_3hop_comp]
    print(f"  Gold set: min={min(gold_sizes)}, max={max(gold_sizes)}, "
          f"mean={np.mean(gold_sizes):.1f}  (of {len(kg_exp5.directors)} directors)")
    q0 = queries_3hop_comp[0]
    print(f"  Sample: {q0.question_text}")
    print(f"  Correct answers: {len(q0.answer_entities)} dirs")
    print(f"  Obs required:    {len(q0.obs_required_facts)} acted_in facts")
    print(f"  Ground rules:    {len(q0.ground_rules)} rules")
    print(f"  Persistent KB:   {len(q0.persistent_facts)} facts")