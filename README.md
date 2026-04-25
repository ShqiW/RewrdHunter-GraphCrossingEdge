# Learning to Untangle: Reinforcement Learning for Graph Layout Optimization

> **Authors:** Shenqian Wen, Zhuocheng Yu

A PPO-based reinforcement learning system that learns to reposition graph nodes to minimize edge crossings, evaluated on the Rome graph benchmark. Our best method, **Sequential Refinement**, achieves **SPC −24.43%** relative to the `neato` force-directed baseline.

---

## Overview

Graph drawing aims to produce visually clear 2D layouts of graphs. Minimizing edge crossings is a key aesthetic criterion, but the crossing count is a piecewise-constant, non-differentiable function of node positions — making it hard to optimize with gradient-based methods.

This project frames layout optimization as a Markov Decision Process (MDP) and trains a PPO agent with a Graph Attention Network (GAT) policy to iteratively reposition nodes. We design and evaluate **five progressively refined approaches**, culminating in a GAT-based iterative refinement policy that achieves substantial crossing reduction over the force-directed baseline.

---

## Key Results

| Method | Mean best-cr ↓ | SPC ↓ | Zero-crossing (%) ↑ |
|---|---|---|---|
| `neato` (baseline) | — | 0.0% | — |
| Discrete PPO | 28.93 | −0.87% | 3.0 |
| Continuous PPO | 28.65 | −2.38% | 4.0 |
| Sequential PPO | 29.25 | +1.11% | 3.0 |
| **Sequential Refinement** | **22.30** | **−24.43%** | **4.0** |
| Node-Select Refinement | 27.43 | −5.20% | 4.0 |

**SPC** (Symmetric Percentage Change) is defined as:

$$\text{SPC} = \frac{\text{cr}_\text{ours} - \text{cr}_\text{ref}}{\max(\text{cr}_\text{ours},\, \text{cr}_\text{ref})}$$

A negative SPC means fewer crossings than the reference. SPC ∈ [−1, 1].

---

## Methods

We develop five methods in order of increasing sophistication. Each is motivated by a specific limitation of its predecessor.

### Method 1: Discrete PPO
Discrete action space of `n × 8 × 3` (node × compass direction × step scale). The policy must jointly decide which node to move and in which direction. Action space scales linearly with graph size.

### Method 2: Continuous PPO
Continuous action space: outputs an `(n × 2)` displacement matrix and moves all nodes simultaneously. More expressive than discrete, but lacks node-level selectivity.

### Method 3: Sequential PPO
A Transformer-based policy that places nodes one-by-one in BFS order. Evaluated in a `neato`-initialized offset variant (SPC +1.11%). Early placement errors compound through subsequent nodes.

### Method 4: Sequential Refinement *(best)*
Starts from a `neato` layout and applies continuous delta moves `(Δx, Δy)` to one node per step, cycling through all nodes in BFS order from a random start. Uses a `GATRefinementPolicy` with 5-dim node features and a patience + reset-to-best mechanism.

![Sequential Refinement Example](0report/seq_refinement_example.png)

*Left: initial `neato` layout (23 crossings). Middle: final layout after refinement (15, Δ=−8). Right: best layout encountered during episode (13, Δ=−10). Red edges = crossings.*

### Method 5: Node-Select Refinement
Extends Method 4 with a learned node-selection head that scores nodes based on GAT embeddings and crossing-count heuristics, plus a crossing-context pooling module. Achieves SPC −5.20% — the discrete selection head receives only a weak gradient signal.

---

## Architecture

### Policy Network (`src/models/`)
- **GAT backbone**: 3 layers, 4 attention heads, hidden dim 128, with edge features (Euclidean edge length)
- **Actor head**: per-node logits (discrete) or Gaussian displacement distribution (continuous/refinement)
- **Critic head**: global mean pooling → scalar value estimate

### Environment (`src/envs/`)
- **State**: node features `[x, y, degree, ...]` + edge features `[edge_length]`; coordinates normalized to `[−1, 1]`
- **Reward**: `w_crossing × ΔCrossings + w_structure × ΔStructure`; optional +10 bonus for zero-crossing layout
- **Structure loss**: SoftmaxRanking (KL divergence, Methods 1–2) or Kamada–Kawai stress (Methods 3–5)
- **Termination**: zero crossings reached, step budget exhausted, or patience exceeded

### Training (`src/train.py`)
Standard PPO: collect rollouts (512 steps) → compute GAE (γ=0.99, λ=0.95) → 4 epochs of mini-batch updates (batch size 512). Checkpoints saved every 10 log intervals.

---

## Installation

```bash
poetry install
# or
bash install.sh
```

> The `controller` package is served from a custom PyPI source (`https://pypi.coredumped.tech/simple/`), configured in `pyproject.toml`.

---

## Usage

### Data Preprocessing

Preprocess the Rome graph dataset into train/test splits:

```bash
poetry run python src/data/rome.py
```

Expects raw GraphML files under `data/rome/` and split lists at `data/train_graph.txt` / `data/test_graph.txt`. Outputs processed `.pt` files to `data/processed/{train,test}/`.

### Training

```bash
# Sequential Refinement (best method)
poetry run python src/main.py \
  --name sequential_refinement \
  --if-train True \
  --total-timesteps 2000000 \
  --graph.use-dataset True

# Discrete PPO
poetry run python src/main.py \
  --name discrete_ppo \
  --if-train True \
  --total-timesteps 500000 \
  --graph.use-dataset True

# With custom hyperparameters (dot notation for nested configs)
poetry run python src/main.py \
  --name sequential_refinement \
  --if-train True \
  --env.max-steps 200 \
  --model.hidden-dim 256 \
  --ppo.lr 3e-4 \
  --total-timesteps 2000000 \
  --save-path checkpoints/my_run
```

### Evaluation / Visualization

```bash
poetry run python src/main.py \
  --name sequential_refinement \
  --if-plot True \
  --save-path checkpoints/my_run
```

### Baseline Comparison

```bash
# Evaluate all methods vs. neato on the test set
poetry run python scripts/22_evaluate_relative.py

# Visualize baseline layouts
poetry run python scripts/21_visualize_baselines.py

# Visualize RL-produced layouts
poetry run python scripts/23_visualize_layouts.py
```

---

## Configuration

Config is composed from nested dataclasses via the `controller`/`tyro` framework. CLI uses dot notation:

| Flag | Description | Default |
|---|---|---|
| `--env.max-steps` | Max steps per episode | `num_nodes × 10` |
| `--env.patience` | Patience steps before reset/truncation | `num_nodes × 3` |
| `--model.hidden-dim` | GAT hidden dimension | `128` |
| `--ppo.lr` | Learning rate | `3e-4` |
| `--graph.use-dataset` | Use Rome dataset (vs. random ER graph) | `False` |
| `--total-timesteps` | Total training timesteps | — |
| `--save-path` | Checkpoint directory | — |

---

## Project Structure

```
├── src/
│   ├── main.py                   # Entry point for all tasks
│   ├── train.py                  # PPO trainer
│   ├── evaluate.py               # Evaluation loop
│   ├── plot.py                   # Layout visualization
│   ├── envs/
│   │   ├── discrete.py           # Method 1: Discrete action env
│   │   ├── continuous.py         # Method 2: Continuous action env
│   │   ├── sequential.py         # Method 3: Sequential placement env
│   │   ├── refinement.py         # Methods 4 & 5: Refinement env
│   │   ├── graph_layout_state.py # Incremental crossing detection
│   │   └── base.py               # Shared env utilities
│   ├── models/
│   │   ├── gnn.py                # DiscreteGNNPolicy (Method 1)
│   │   ├── continuous_gnn.py     # ContinuousGNNPolicy (Method 2)
│   │   ├── transformer_policy.py # TransformerPolicy (Method 3)
│   │   ├── gat_refinement_policy.py  # GATRefinementPolicy (Method 4)
│   │   └── node_select_gnn.py    # NodeSelectGNNPolicy (Method 5)
│   ├── losses/
│   │   ├── xing.py               # Edge crossing loss (hard & soft)
│   │   ├── stress.py             # Kamada-Kawai stress loss
│   │   └── softmax_ranking.py    # SoftmaxRanking structure loss
│   ├── data/
│   │   └── rome.py               # RomeDataset loader & preprocessor
│   └── tasks/
│       └── base.py               # Config dataclasses
├── scripts/                      # Analysis and visualization scripts
├── data/
│   ├── rome/                     # Raw GraphML files
│   ├── processed/
│   │   ├── train/                # Preprocessed train graphs (.pt)
│   │   └── test/                 # Preprocessed test graphs (.pt)
│   ├── train_graph.txt
│   └── test_graph.txt
├── 0report/
│   └── final_report.tex          # Full technical report
└── pyproject.toml
```

---

## Hyperparameters (Sequential Refinement)

| Hyperparameter | Value |
|---|---|
| Learning rate | 3×10⁻⁴ |
| Discount γ | 0.99 |
| GAE λ | 0.95 |
| Clip ε | 0.2 |
| PPO update epochs | 4 |
| Batch size | 512 |
| Rollout steps | 512 |
| Entropy coefficient | 0.05 |
| Hidden dimension | 128 |
| GAT layers / heads | 3 / 4 |
| Structure weight w_s | 0.3 |
| Patience | 100 steps |
| Reset-to-best | enabled |

---

## Dataset

The **Rome graph collection** is a standard benchmark in graph drawing, containing real-world graphs from circuit schematics, biological networks, and social graphs. Each graph is preprocessed to include:
- Edge index and per-node degree
- All-pairs shortest-path distances
- Precomputed SoftmaxRanking distance distribution for the structure loss

A new graph is sampled from the training set at the start of each episode, so the agent must generalize across different graph topologies.
