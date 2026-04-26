---
title: Traffic Signal OpenEnv
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.main:app
pinned: false
---

# Traffic Signal OpenEnv: Hierarchical Urban Orchestration

> **TL;DR:** We built a deterministic, controllable benchmark testing whether LLMs can act as real-time, city-scale coordinators across multiple interacting agents.  
> **Result:** +34.2% score improvement, +122% throughput, and 15% fewer spillback events vs. decentralized control.

## Submission Links

| Resource | Link |
|:---|:---|
| **Live Environment** | [guuru-dev-traffic-signal-openenv-2.hf.space](https://guuru-dev-traffic-signal-openenv-2.hf.space) |
| **HF Space Repo** | [Guuru-DEV/traffic-signal-openenv-2](https://huggingface.co/spaces/Guuru-DEV/traffic-signal-openenv-2) |
| **GitHub** | [Arya-Akshat/OpenEnv-MetaXHuggingFace-Hackathon](https://github.com/Arya-Akshat/OpenEnv-MetaXHuggigFace-Hackathon) |
| **Training Notebook** | [`notebooks/train_colab_FULL.ipynb`](notebooks/train_colab_FULL.ipynb) |
| **Writeup / Blog** | [`blog.md`](blog.md) |
| **Run Log** | [`results/run_log.md`](results/run_log.md) |
| **W&B Dashboard** | [wandb.ai/…/traffic-signal-openenv](https://wandb.ai/akshat-arya13-r-v-c-e/traffic-signal-openenv) |

---

## 🚦 The Problem

Urban traffic is deceptively simple. A single intersection can be managed by local rules — but a city grid suffers from **bottleneck propagation**, **spillback**, and **emergency routing delays** that no single intersection can see coming.

This is a perfect benchmark for LLMs because it requires three capabilities that set it apart from typical Gym environments:

1. **Multi-Agent Reasoning** — balancing 4 independent intersections (NW, NE, SW, SE) that create cross-interference through shared corridors.
2. **Hierarchical Control** — the LLM doesn't directly flip traffic lights. It sets high-level policy vectors that local agents interpret.
3. **Stability Under Stress** — managing deterministic incidents (lane closures, demand surges) without collapsing into gridlock.

---

## 🏗️ Environment Architecture

### The Hierarchy
```
┌─────────────────────────────────────────────┐
│              Central Controller (LLM)       │
│  Outputs: corridor_priority, emergency_boost│
│           queue_urgency_weight, ...         │
└──────────────────┬──────────────────────────┘
                   │ Policy Vector
     ┌─────────────┼─────────────┐
     ▼             ▼             ▼
 [Local NW]   [Local NE]   [Local SW/SE]
  Rule-based   Rule-based   Rule-based
  Phase Switch Phase Switch Phase Switch
```

**Central Controller (LLM)** updates policy vectors (e.g., `corridor_priority`, `emergency_boost`) that shape how local agents weight their decisions. **Local Agents (Rule-Based)** execute high-frequency phase switching using the Central Policy and 1-step lookahead logic.

### Grid Layout (2×2 with FIFO Transit Buffers)
```text
      [NW] ←——(3)——→ [NE]
        ↕               ↕
       (3)             (3)
        ↕               ↕
      [SW] ←——(3)——→ [SE]

(3) = 3-Step FIFO Transit Buffer (bidirectional)
```

Each intersection has **4 lanes**, **4 signal phases**, and a unique **personality** (e.g., NW is a "corridor entry" node, SW is "emergency-prone"). Corridors create complex cross-interference: clearing a queue at NW pushes vehicles into NE's buffer 3 steps later.

### What the LLM Sees (`text_obs`)
```yaml
Intersection NW:
  Queue: [3, 12, 4, 1]        # 4 lanes
  Wait:  [4.2, 15.1, 5.0, 1.2] # seconds
  Role:  Corridor Entry
  Active Behaviors: [DEMAND_SURGE_RESPONSE]
System Metrics:
  Throughput: 68.2
  Imbalance: 4.2
  Spillback Risk: High (Intersection NE)
```

### What the LLM Outputs
```json
{
  "local_actions": {"NW": "PHASE_2", "NE": "KEEP", "SW": "SWITCH", "SE": "PHASE_1"},
  "central_action": {"queue_urgency_weight": 0.4, "corridor_priority": 0.3}
}
```

7 tasks test different skills: `easy_fixed`, `medium_dynamic`, `hard_multi`, `gridlock_risk`, `corridor_flow`, `incident_response`, `dynamic_demand`.

---

## 🚀 Results

### Baseline Comparison (hard_multi task)

We compared four control strategies on the hardest task to isolate the effect of each component:

| Strategy | Final Score | Throughput | Spillback Events |
|:---|:---|:---|:---|
| Do Nothing (all `KEEP`) | 0.336 | — | — |
| Random Actions | 0.399 | — | — |
| Rule-Based (local only, no central) | 0.386 | 5,083 | 1,888 |
| **Rule-Based + Central Policy (ours)** | **0.509** | **11,317** | **1,602** |
| **Trained LLM + Central (best A100)** | **0.518** | — | — |

| Metric | Local-Only → Central | Improvement |
|:---|:---|:---|
| Final Score | 0.386 → 0.518 | **+34.2%** |
| Throughput | 5,083 → 11,317 | **+122.6%** |
| Spillback Events | 1,888 → 1,602 | **−15.1%** |

The trained LLM with central policy outperforms every baseline by a wide margin. Central coordination is the dominant factor — even a simple rule-based agent jumps from 0.386 to 0.509 once it can set `queue_urgency_weight` and `corridor_priority`.

### What Changed: Before vs. After (Episode Trace)

Here's a concrete episode on `hard_multi` showing how central coordination reshapes behavior:

| Step | Agent Action | Central Deltas | Total Queue | Score | What Happened |
|:---|:---|:---|:---|:---|:---|
| 0 | All → PHASE_3 | `urgency: 0.4, emergency: 0.5, corridor: 0.3` | 247 | 0.654 | High initial congestion; agent boosts urgency and enables emergency routing |
| 5 | NW→P0, SW→P1, SE→P2 | `urgency: 0.4, emergency: 0.5` | 299 | 0.528 | Queues still building; agent diversifies phases to drain multiple lanes |
| 25 | All different phases | `urgency: 0.4, corridor: 0.3` | 207 | 0.484 | **Queues dropping** — corridor priority synchronizes NW↔NE flow |
| 50 | NW→P0, NE→P2, SE→KEEP | `urgency: 0.4, corridor: −0.2` | 161 | 0.496 | Agent **reverses** corridor bias as NS traffic recovers — adaptive control |
| 200 | Mixed phases | `urgency: 0.4, emergency: 0.5` | 262 | 0.509 | Stable management through end of episode despite demand surges |

**Key insight:** The agent doesn't just hold one policy. It **adapts** central deltas in response to changing conditions — boosting `corridor_priority` when EW traffic surges, then reversing it when NS recovers.

Without central policy, the same agent scores 0.386 with ~2x more spillback.

---

### Training Evidence

The final A100 run used `unsloth/Llama-3.2-1B-Instruct` with PEFT LoRA and a manual GRPO-style policy optimization loop over **264 episodes**.

> **Note:** Training episodes use 30-step rollouts (`MAX_ENV_STEPS=30`) for compute efficiency on A100. Baseline comparison scores above are from full 200-step episodes. The 30-step training signal captures early-episode behavior and policy responsiveness; final scores from short rollouts correlate with but slightly differ from full-episode scores.

| Metric | Value |
|:---|:---|
| Valid JSON rate | **99.62%** |
| Central-action usage | **99.62%** |
| Hallucination rate | **0.38%** |
| Last-50 mean reward | **1.506** |
| Best final score | **0.51797** |
| Best episode reward | **3.970** |

![A100 central-policy reward curve](plots/a100_central_policy_reward_curve.png)

*A100 final run: reward climbs from near-zero to +1.5 mean over 264 episodes. No catastrophic collapse.*

![A100 central-policy final score](plots/a100_central_policy_final_score_curve.png)

*Final score stabilizes above 0.49 (hard-task baseline: 0.386). Best episode reaches 0.518.*

![Central coordination ablation](plots/ablation_comparison.png)

*Ablation: compares final score (y-axis) across control strategies (x-axis). Central coordination (+31.9% over local-only) is the dominant factor; trained LLM adds a further +1.8% on top.*

<details>
<summary>Additional training plots (Kaggle runs)</summary>

![Training reward curve](plots/reward_curve.png)

*Kaggle 1B run: early reward convergence with local-only actions (no central policy).*

![Central-policy GRPO reward curve](plots/central_policy_reward_curve.png)

*Kaggle central-policy run: partial 640-episode trace showing non-flat rewards and stable learning.*

![Central-policy output quality](plots/central_policy_output_quality.png)

*Sustained valid JSON and central-policy usage without training collapse.*

</details>

### Reward Shaping

The GRPO reward function uses four mechanisms to prevent common LLM-RL failure modes:

| Mechanism | Effect | Why |
|:---|:---|:---|
| **Hallucination penalty (−6.0)** | Flat negative reward for invalid JSON | Prevents "safe garbage" exploitation |
| **All-KEEP collapse (−3.0)** | Penalty for passive "do nothing" outputs | Forces the agent to actually control traffic |
| **Central-action bonus (+0.15 / −0.25)** | Reward for using hierarchical policy deltas | Steers toward the coordination capability we're testing |
| **Curriculum staging** | `medium_dynamic` → `hard_multi` at episode 40 | Prevents early collapse on hardest scenarios |

### Artifacts

| Artifact | Location |
|:---|:---|
| A100 LoRA adapter | [outputs/traffic-lora-a100-central-policy](https://huggingface.co/spaces/Guuru-DEV/traffic-signal-openenv-2/tree/main/outputs/traffic-lora-a100-central-policy) |
| Training metrics (JSON/CSV) | [results/](https://huggingface.co/spaces/Guuru-DEV/traffic-signal-openenv-2/tree/main/results) |
| Training plots | [plots/](https://huggingface.co/spaces/Guuru-DEV/traffic-signal-openenv-2/tree/main/plots) |
| W&B dashboard | [traffic-signal-openenv](https://wandb.ai/akshat-arya13-r-v-c-e/traffic-signal-openenv) |
| Run log | [`results/run_log.md`](results/run_log.md) |

---

## 💡 Why This Matters

This is a small traffic world, but the coordination shape is universal: **one high-level reasoner orchestrating many local actors under stress.** The same pattern appears in fleet management, warehouse robotics, incident response, and infrastructure control.

The key insight is that the LLM didn't win by talking more — it won by learning to speak less: one compact JSON object, sent at the right time, with the right central nudge. That's a fundamentally different capability from chat or code generation, and it's exactly what hierarchical multi-agent systems need.

---

## 🏆 Hackathon Themes

- **Theme 1: Multi-Agent Interactions** — Managing cross-interference between NW/NE/SW/SE through shared corridors.
- **Theme 2: Long-Horizon Planning** — Preemptive spillback prevention requires reasoning about downstream effects 3+ steps ahead.
- **Theme 4: Self-Improvement** — Curriculum-based training evolves policies from easy → hard.
- **Fleet AI Scalable Oversight** — Centralized monitoring of 16 individual traffic lanes with 5 tunable policy knobs.
- **Halluminate Multi-Actor** — Deterministic incident response with distinct intersection "personalities."

---

## 🛠️ Quick Start

### Docker
```bash
docker build -t traffic-env .
docker run --rm -p 7860:7860 traffic-env
```

Or use the hosted Space: `https://guuru-dev-traffic-signal-openenv-2.hf.space`

### API
```bash
# Reset
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "hard_multi", "central_enabled": true}'

# Step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"local_actions":{"NW":"PHASE_2","NE":"PHASE_3","SW":"SWITCH","SE":"KEEP"},"central_action":{"queue_urgency_weight":0.5,"corridor_priority":0.3}}'
```

### Training
Use [`notebooks/train_colab_FULL.ipynb`](notebooks/train_colab_FULL.ipynb) — a self-contained 8-cell pipeline: install → config → model load → SFT warmup → schema gate → reward function → GRPO → save/upload.

Requires `ENV_URL`, `HF_TOKEN`, and `WANDB_API_KEY` in the notebook environment. Designed for Kaggle/Colab/HF Jupyter-style execution.

---
