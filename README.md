# SELLNSIA — Resource-Aware Neuro-Symbolic KG/KB-QA via Subexponential Linear Logic (SELL)

This repository contains a **reproducible experiment notebook** implementing and evaluating a **resource-aware neuro-symbolic** approach to **Knowledge Graph / Knowledge Base Question Answering (KG/KB-QA)** based on **Subexponential Linear Logic (SELL)**.

Main artifact: **`SELLExpNSy.ipynb`**

The notebook is designed to be **self-contained**: it implements the SELL prover directly in Python and does **not** rely on external theorem provers.

---

## What’s inside the notebook (`SELLExpNSy.ipynb`)

The notebook provides an end-to-end experimental pipeline:

### 1) SELL Prover (focused proof search)
A focused proof-search engine for a grounded SELL-Horn fragment using **four subexponential labels**:

| Label | Zone | Reusable? | Intended meaning |
|------|------|-----------|------------------|
| `kg`  | Ψ | Yes | Persistent KG facts |
| `rl`  | Ψ | Yes | Persistent ground rules |
| `obs` | Γ | No  | Consumable observations (evidence) |
| `bud` | Γ | No  | Budget tokens |

It implements the multiplicative rules (⊗L, ⊗R, ⊸L) and the labeled dereliction discipline **Use_u / Use_ℓ** described in the notebook text.

### 2) Trace / certificate verification
An independent multiset-rewriting checker (Algorithm 2 in the notebook) that replays proof traces to validate:
- correct premise availability,
- correct consumption of linear resources (`obs`),
- correct enforcement of budget (`bud`).

### 3) Controlled benchmark construction (MetaQA-style movie KG)
A synthetic movie-domain KG (movies, actors, directors, genres, writers) with automatically generated:
- **1-hop** queries
- **2-hop** queries
- **3-hop** queries

The setup supports:
- configurable **noise injection** into observations,
- varying **budget constraints**,
- controlled separation between **persistent facts** (trusted backbone) and **observation-only facts** (retrieved evidence).

### 4) Neural scorer (TransE-style)
A lightweight TransE-like embedding scorer used to rank candidate triples (observations) per query.

### 5) Baselines & ablations
The notebook compares multiple reasoning/resource-management variants, including:
- neural-only
- classical LP (all facts persistent)
- SELL without budget
- SELL with observations treated as persistent
- full SELL (consumable observations + enforced budget)

### 6) Publication-ready outputs
Generates figures (matplotlib) and LaTeX-formatted tables and exports them to `./results/`.

---

## Reproducibility

- The notebook sets `SEED = 42` for deterministic behavior of stochastic components (as far as the environment allows).
- Outputs are written to a local folder named **`results/`**.

---

## Requirements

The notebook uses standard scientific Python packages:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `tabulate`
- `scikit-learn`
- `tqdm`

The first cells attempt to import these and install missing ones.

---

## How to run (local)

### Option A — Jupyter Notebook
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install notebook
jupyter notebook
```

Open `SELLExpNSy.ipynb` and run **Kernel → Restart & Run All**.

### Option B — VS Code
- Open the repository in VS Code
- Install the Python + Jupyter extensions
- Open `SELLExpNSy.ipynb` and run all cells in order

---

## Output

After execution, you should see:
- `results/` — generated tables and figures (created automatically)

> Tip: If you plan to keep the repo clean, consider adding `results/` to `.gitignore` since outputs are reproducible.

---

## Experimental design (high-level)

- **Persistent context (Ψ):** trusted, reusable facts/rules (e.g., KG backbone and grounded rules).
- **Linear context (Γ):** consumable evidence (`obs`) plus budget tokens (`bud`).
- **Budget:** each budgeted rule application consumes one token; this controls proof cost and prevents unconstrained chaining.

This setup enables controlled evaluation of:
- robustness to noisy evidence,
- multi-hop reasoning sensitivity (1/2/3-hop),
- resource-aware behavior (consumption vs reuse),
- budget/accuracy trade-offs via proof traces.

---

## Reference / description

If you cite or describe this repository, you can summarize it as:

A **resource-aware neuro-symbolic KG/KB-QA** experimental pipeline based on **Subexponential Linear Logic (SELL)**, including a focused SELL-Horn prover with verifiable proof traces, a controlled MetaQA-style benchmark with noise and budgets, and a TransE-style neural triple scorer, plus baselines and ablations.

---

## Author

GitHub user: **@cero1979**
