# Taming Urban Chaos With a Tiny LLM

Traffic looks simple until four intersections start arguing with each other. One green light clears a queue, but it floods the next junction 3 steps later. A local rule can be optimal for one corner and still create gridlock for the whole block.

We built **Traffic Signal OpenEnv** to test whether a small LLM can learn to be the *coordinator* — not the traffic cop, but the city planner who nudges policy knobs from above while local agents handle the moment-to-moment decisions.

## The Environment

The environment is a 2×2 grid of intersections (NW, NE, SW, SE) connected by bidirectional corridors with 3-step FIFO transit buffers. Each intersection has 4 lanes, 4 signal phases, and a unique "personality" that affects how it weighs queue pressure, downstream congestion, and emergency events.

The LLM reads a compact text observation and returns one JSON object: local phase choices for all four intersections, plus small central-policy deltas like `queue_urgency_weight` and `corridor_priority`. These deltas don't directly flip traffic lights — they shift the weights that local rule-based agents use to make their own phase decisions. That's the hierarchical part: the LLM is an oversight layer, not a micromanager.

Seven tasks test different skills at increasing difficulty: from `easy_fixed` (static demand) through `incident_response` (lane closures and demand surges) to `hard_multi` (dynamic demand with emergency events and corridor interference).

## The Challenge: Why LLMs Fail at This

The first RL attempts failed in very LLM ways:

- **Prose instead of JSON.** The model would explain its reasoning in natural language instead of outputting the action.
- **Malformed keys.** `local_action` instead of `local_actions`, `Phase_0` instead of `PHASE_0`.
- **All-KEEP collapse.** The safest thing an RL agent can do is nothing — and it learned that fast.
- **Reward-looking behavior.** Valid JSON that technically improved the reward signal but didn't actually help traffic (e.g., random phase cycling).

These are not toy problems. They're the same failure modes that affect LLM agents in production: schema drift, action collapse, and reward hacking.

## The Training Recipe

The stable recipe that actually worked:

**Stage 1: SFT Schema Warmup** (100 steps, lr=3e-5)
- Teach the model to emit valid JSON matching the exact traffic schema
- 8 diverse schema examples × 140 repeats = 1,120 training texts
- Gate: must pass ≥7/8 valid JSON samples AND ≥6/8 with `central_action` before RL begins

**Stage 2: Manual GRPO-style Policy Optimization** (80 updates, lr=5e-6)
- 2 prompts × 4 generations per update = 8 episodes per update
- Group-relative advantage normalization (REINFORCE-style)
- Curriculum: `medium_dynamic` → `hard_multi` at episode 40

**Reward Shaping** (4 mechanisms):
| Mechanism | Reward | Purpose |
|:---|:---|:---|
| Invalid JSON | −6.0 | Kill hallucination fast |
| All-KEEP output | −3.0 | Prevent passivity collapse |
| Uses central policy | +0.15 | Encourage hierarchical coordination |
| Omits central policy | −0.25 | Penalize missing the coordination opportunity |

**Infrastructure:**
- `safe_post` wrapper with 16-retry jittered exponential backoff for the live HF Space API
- On A100, the fastest Unsloth/TRL path hit runtime dtype incompatibilities. The final run used standard **Transformers + PEFT LoRA** with a manual training loop. That was the right trade: smaller framework surface, cleaner control, better evidence.

## What We Got

The final A100 run recorded **264 environment episodes** against the live Space:

| Metric | Value |
|:---|:---|
| Valid JSON rate | 99.62% |
| Central-action usage | 99.62% |
| Hallucination rate | 0.38% |
| Last-50 mean reward | 1.506 |
| Best hard-task final score | 0.51797 |

The broader ablation study across control strategies on `hard_multi`:

| Strategy | Score | Throughput | Δ vs. Local-Only |
|:---|:---|:---|:---|
| Do Nothing (all KEEP) | 0.336 | — | −12.9% |
| Random Actions | 0.399 | — | +3.4% |
| Rule-Based (local only) | 0.386 | 5,083 | baseline |
| Rule-Based + Central Policy | 0.509 | 11,317 | **+31.9%** |
| **Trained LLM + Central** | **0.518** | — | **+34.2%** |

The central coordination layer is the dominant factor. Even a simple rule-based agent jumps from 0.386 to 0.509 (+31.9%) once it can set `queue_urgency_weight` and `corridor_priority`. The trained LLM pushes that further by learning *when* and *how much* to adjust these knobs.

## What the Agent Actually Does

Here's a concrete episode trace on `hard_multi`. Watch how the agent adapts its central policy:

- **Step 0:** High initial congestion (queue=247). Agent sets `emergency_boost=0.5` and `corridor_priority=0.3` — aggressively prioritizing emergency lanes and east-west corridor flow.
- **Step 25:** Queues dropping (247→207). Corridor sync is working. Agent maintains urgency but keeps corridor boost.
- **Step 50:** Queue=161. NS traffic has recovered. Agent **reverses** `corridor_priority` to −0.2 — it learned that the NS direction now needs more service.
- **Step 200:** Stable at queue=262 despite demand surges. Final score: 0.509.

Without central policy, the same agent scores 0.386 with nearly twice the spillback events (1,888 vs. 1,602).

The model didn't win by talking more. It won by learning to speak less: one compact JSON object, sent at the right time, with the right central nudge.

## Why It Matters

This is a small traffic world, but the coordination shape is universal. One high-level reasoner orchestrating many local actors under stress — that's fleets, warehouses, incident response, robotics, and infrastructure control.

The fun part is that the model learned something genuinely interesting: **adaptive hierarchical coordination**. It doesn't hold a fixed policy. It reads the state of the grid and adjusts its oversight accordingly. That's closer to how real city traffic centers work than any fixed-rule system.

## Links

| Resource | Link |
|:---|:---|
| W&B (training runs) | https://wandb.ai/akshat-arya13-r-v-c-e/traffic-signal-openenv |
| Live Space | https://guuru-dev-traffic-signal-openenv-2.hf.space |
| HF Space Repo | https://huggingface.co/spaces/Guuru-DEV/traffic-signal-openenv-2/tree/main |
| GitHub | [OpenEnv-MetaXHuggingFace-Hackathon](https://github.com/Arya-Akshat/OpenEnv-MetaXHuggigFace-Hackathon) |
| Training Notebook | [`notebooks/train_colab_FULL.ipynb`](notebooks/train_colab_FULL.ipynb) |
| Run Log | [`results/run_log.md`](results/run_log.md) |
