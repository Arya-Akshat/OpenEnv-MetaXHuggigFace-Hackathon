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
> **Result:** +36.2% performance improvement and ~40% faster recovery from disruptions vs. decentralized control.

**A Deterministic LLM Benchmark for Multi-Agent Traffic Control**

Traffic Signal OpenEnv is a high-fidelity, hierarchical traffic-light orchestration platform. It is designed to test an LLM's ability to act as a **Central Controller**, managing grid-level policy vectors to optimize flow across multiple local agents.

## Submission Links

- **Live environment Space**: [guuru-dev-traffic-signal-openenv-2.hf.space](https://guuru-dev-traffic-signal-openenv-2.hf.space)
- **Space repository and artifacts**: [Guuru-DEV/traffic-signal-openenv-2](https://huggingface.co/spaces/Guuru-DEV/traffic-signal-openenv-2)
- **GitHub repository**: [Arya-Akshat/OpenEnv-MetaXHuggingFace-Hackathon](https://github.com/Arya-Akshat/OpenEnv-MetaXHuggigFace-Hackathon)
- **Canonical training notebook**: [`notebooks/train_colab_FULL.ipynb`](notebooks/train_colab_FULL.ipynb)
- **Writeup**: [`blog.md`](blog.md) — Explains our methodology and core design decisions.
- **Run log**: [`results/run_log.md`](results/run_log.md) — Summarizes A100 training runs, metrics, and diagnostics.
- **Weights & Biases (live training metrics)**: [wandb.ai/.../traffic-signal-openenv](https://wandb.ai/akshat-arya13-r-v-c-e/traffic-signal-openenv) — notebook runs with `WANDB_API_KEY` sync here (episodes, rewards, loss, artifacts).

---

## 🚦 The Problem: The Hidden Cost of Uncoordinated Flow
Urban traffic is a deceptively simple problem. While a single intersection can be managed by local rules, a city grid suffers from **bottleneck propagation**, **spillback**, and **emergency routing delays**. Traditional systems lack the long-horizon reasoning required to preemptively throttle flow or synchronize "Green Waves" across corridors.

This is a perfect benchmark for LLMs because it requires:
1.  **Multi-Agent Reasoning**: Balancing 4 independent intersections NW, NE, SW, SE.
2.  **Chain-of-Thought Decision Making**: Explaining "why" a policy shift is necessary.
3.  **Stability Under Stress**: Managing deterministic incidents (closures, surges) without collapsing into gridlock.

---

## 🏗️ Environment Architecture

### The Hierarchy
- **Central Controller (LLM)**: Updates policy vectors (e.g., `corridor_priority`, `emergency_boost`) every $N$ steps.
- **Local Agents (Rule-Based)**: Execute high-frequency phase switching based on the Central Policy and 1-step lookahead logic.

### Grid Layout (2x2)
```text
      [NW] <---(3)--- [NE]
        |               |
       (3)             (3)
        |               |
      [SW] ---(3)---> [SE]
      
(3) = 3-Step FIFO Transit Buffer
```

**Note:** Corridors feature bidirectional traffic flow, creating complex cross-interference at intersections.

### The `text_obs` Interface
The environment provides a structured, YAML-like observation designed specifically for LLM ingestion:
```yaml
Intersection NW:
  Queue: [3, 12, 4, 1]
  Wait: [4.2, 15.1, 5.0, 1.2]
  Role: Corridor Entry
  Active Behaviors: [DEMAND_SURGE_RESPONSE]
System Metrics:
  Throughput: 68.2
  Imbalance: 4.2
  Spillback Risk: High (Intersection NE)
```

---

## 🚀 Results: Proven Gains
Hierarchical central coordination closes a large gap versus local-only control:

| Condition | Final Score (max: 1.0) | Throughput |
| :--- | :--- | :--- |
| Local-only (baseline) | 0.380 | 51.2 |
| Central LLM (ours) | 0.518 | 68.2 |
| **Δ Improvement** | **+36.2%** | **+33.2%** |

On medium-difficulty tasks, central coordination improved `final_score` by about **+23%** vs. local-only control (see `results/run_log.md`).

### Final Training Artifacts
The final training flow uses a two-stage pipeline:
- **SFT schema warmup** to teach strict JSON action formatting.
- **Central-policy policy optimization** to tune traffic-control actions only after schema validation passes.

Reward shaping during GRPO uses four mechanisms to prevent common LLM-RL failure modes:
1. **Hallucination penalty (−6.0):** If the model emits anything other than valid schema JSON, the episode receives a flat −6.0 reward, preventing "safe garbage" exploitation.
2. **All-KEEP collapse penalty (−3.0):** If every intersection action is `KEEP`, the model is penalized for passivity.
3. **Central-action bonus (+0.15 / −0.25):** Episodes that include learned `central_action` deltas receive a small bonus; omitting them incurs a penalty. This steers the model toward hierarchical coordination.
4. **Curriculum staging:** Training begins on `medium_dynamic` tasks and graduates to `hard_multi` at episode 40, preventing early policy collapse on the hardest scenarios.

For the final A100 run, we engineered a custom Transformers + PEFT LoRA pipeline to bypass Unsloth/TRL constraints, achieving maximum training stability. Our compute-constrained A100 run showed strong early convergence signals over 264 episodes, achieving **99.62% valid JSON actions**, **99.62% central-action usage**, **0.38% hallucination rate**, **1.506 last-50 mean reward**, and a **0.51797 best final score**.

Generated artifacts are available in the live Space repository:
- **A100 LoRA adapter**: [`outputs/traffic-lora-a100-central-policy`](https://huggingface.co/spaces/Guuru-DEV/traffic-signal-openenv-2/tree/main/outputs/traffic-lora-a100-central-policy)
- **Training plots**: [`plots`](https://huggingface.co/spaces/Guuru-DEV/traffic-signal-openenv-2/tree/main/plots)
- **Training metrics**: [`results`](https://huggingface.co/spaces/Guuru-DEV/traffic-signal-openenv-2/tree/main/results)
- **Weights & Biases**: [traffic-signal-openenv](https://wandb.ai/akshat-arya13-r-v-c-e/traffic-signal-openenv) (same project as Submission Links)
- **Run log**: [`results/run_log.md`](results/run_log.md)

Generated plots include the final A100 central-policy run (`a100_central_policy_reward_curve.png`, `a100_central_policy_final_score_curve.png`, `a100_central_policy_output_quality.png`), the Kaggle central-policy run, the ablation comparison, and earlier reward/score diagnostics.

![Training reward curve](plots/reward_curve.png)

*Shows stable RL policy improvement with no catastrophic collapse.*

![A100 central-policy reward curve](plots/a100_central_policy_reward_curve.png)

*Shows stable RL policy improvement with no catastrophic collapse.*

![A100 central-policy final score](plots/a100_central_policy_final_score_curve.png)

*Demonstrates consistent system-wide gains under central coordination.*

![Central-policy GRPO reward curve](plots/central_policy_reward_curve.png)

*Shows stable RL policy improvement with no catastrophic collapse.*

![Central-policy output quality](plots/central_policy_output_quality.png)

*Highlights sustained valid JSON and central-policy usage without training collapse.*

![Central coordination ablation](plots/ablation_comparison.png)

*Confirms central coordination is the primary driver of throughput.*

---

## 🏆 Hackathon Themes & Sub-themes
- **Theme 1: Multi-Agent Interactions**: Managing the complex interplay between NW/NE/SW/SE.
- **Theme 2: Long-Horizon Planning**: Preemptively managing downstream spillback risks.
- **Theme 4: Self-Improvement**: Using the `--curriculum` runner to evolve policies.
- **Fleet AI Scalable Oversight**: Centralized monitoring of 16 individual traffic lanes.
- **Halluminate Multi-Actor**: Deterministic incident response requiring distinct "personalities" per intersection.

---

## 🛠️ Quick Start

Replace `localhost:7860` with your Space URL (for example `https://guuru-dev-traffic-signal-openenv-2.hf.space`) when running against the hosted environment instead of a local container.

### Docker (Recommended)
```bash
docker build -t traffic-env .
docker run --rm -p 7860:7860 traffic-env
```

### Local CLI
```bash
# Reset with specific task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "hard_multi", "central_enabled": true}'

# Execute step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"local_actions":{"NW":"PHASE_2","NE":"PHASE_3","SW":"SWITCH","SE":"KEEP"},"central_action":{"queue_urgency_weight":0.5,"corridor_priority":0.3}}'
```

### Example Outcome

After ~50 steps of Central LLM execution:

- Increased throughput across all local intersections.
- Massively reduced spillback risk on critical corridors.
- Emergent corridor synchronization (intelligent "green waves").

### Training
Use `notebooks/train_colab_FULL.ipynb` for the self-contained final training flow. It uses a small 1B model, PEFT LoRA, SFT schema warmup, schema validation, a manual GRPO-style policy loop, W&B tracking, graceful API retries, and automatic artifact upload.

The notebook is designed for Kaggle/Colab/HF Jupyter-style execution: it expects `HF_TOKEN`, `WANDB_API_KEY`, and `ENV_URL` in the notebook environment and does not clone the repository during training. `training/train.py` remains as a lightweight script entrypoint, but the notebook is the canonical submission training artifact.

---

