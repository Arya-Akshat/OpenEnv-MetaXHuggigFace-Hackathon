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

**A Deterministic LLM Benchmark for Multi-Agent Traffic Control**

Traffic Signal OpenEnv is a high-fidelity, hierarchical traffic-light orchestration platform. It is designed to test an LLM's ability to act as a **Central Controller**, managing grid-level policy vectors to optimize flow across multiple local agents.

Live deployment: [Hugging Face Space](https://guuru-dev-traffic-signal-openenv-2.hf.space)

Writeup: [`blog.md`](blog.md)

---

## 🚦 The Problem: The Hidden Cost of Uncoordinated Flow
Urban traffic is a "deceptively simple" problem. While a single intersection can be managed by local rules, a city grid suffers from **bottleneck propagation**, **spillback**, and **emergency routing delays**. Traditional systems lack the long-horizon reasoning required to preemptively throttle flow or synchronize "Green Waves" across corridors.

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
Through our **11-Phase Refactor**, we achieved a massive performance gap demonstrating the power of hierarchical oversight:
- **Hard Multi-Task Improvement**: **+36.2%** increase in `final_score` with Central Coordination enabled.
- **Medium Task Improvement**: **+23%**.
- **Recovery Efficiency**: The system recovers from incidents (lane closures) 40% faster under central guidance.

### Final Training Artifacts
The final training flow uses a two-stage pipeline:
- **SFT schema warmup** to teach strict JSON action formatting.
- **Gated GRPO environment tuning** to optimize traffic-control behavior only after schema validation passes.

Generated artifacts are available in the live Space repository:
- **LoRA adapter**: [`outputs/traffic-lora`](https://huggingface.co/spaces/Guuru-DEV/traffic-signal-openenv-2/tree/main/outputs/traffic-lora)
- **Training plots**: [`plots`](https://huggingface.co/spaces/Guuru-DEV/traffic-signal-openenv-2/tree/main/plots)
- **Training metrics**: [`results`](https://huggingface.co/spaces/Guuru-DEV/traffic-signal-openenv-2/tree/main/results)
- **W&B project**: [traffic-signal-openenv](https://wandb.ai/akshat-arya13-r-v-c-e/traffic-signal-openenv)
- **Run log**: [`results/run_log.md`](results/run_log.md)

Generated plots include `ablation_comparison.png`, `reward_breakdown.png`, `final_score_curve.png`, and `reward_curve.png`.

![Training reward curve](plots/reward_curve.png)

![Central coordination ablation](plots/ablation_comparison.png)

---

## 🏆 Hackathon Themes & Sub-themes
- **Theme 1: Multi-Agent Interactions**: Managing the complex interplay between NW/NE/SW/SE.
- **Theme 2: Long-Horizon Planning**: Preemptively managing downstream spillback risks.
- **Theme 4: Self-Improvement**: Using the `--curriculum` runner to evolve policies.
- **Fleet AI Scalable Oversight**: Centralized monitoring of 16 individual traffic lanes.
- **Halluminate Multi-Actor**: Deterministic incident response requiring distinct "personalities" per intersection.

---

## 🛠️ Quick Start

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
  -d '{"task_id": "hard_multi"}'

# Execute step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": "PHASE_0"}'
```

### Training
Use `notebooks/train_colab_FULL.ipynb` for the full SFT + GRPO run, or `training/train.py` for the script-based training entrypoint.

The full notebook is self-contained for Kaggle/Colab-style execution: it expects `HF_TOKEN`, `WANDB_API_KEY`, and `ENV_URL` in the notebook environment and does not clone the repository during training.

---

