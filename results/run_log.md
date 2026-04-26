# Training Run Log

This file tracks real training attempts and generated evidence for the hackathon submission.

## Current Evidence

- Environment: Hugging Face Space `Guuru-DEV/traffic-signal-openenv-2`
- Model family: `unsloth/Llama-3.2-1B-Instruct`
- Training stack: SFT schema warmup + central-policy policy optimization
- Monitoring: Weights & Biases project [`traffic-signal-openenv`](https://wandb.ai/akshat-arya13-r-v-c-e/traffic-signal-openenv) (all synced runs; filter by run name, e.g. `openenv-a100-central-policy-1b-*`)
- Local plots committed in `plots/`
- Large adapter artifacts are stored on the Hugging Face Space, not in Git.

## Runs

### 1B Schema Warmup + Local-Action GRPO

- Status: completed partially, then Kaggle notebook crashed after extended running.
- Result: SFT schema warmup was strong (`5/5` valid JSON samples after warmup).
- GRPO issue found: raw prompts caused prose/hallucination collapse until the prompt path was switched to the same chat-template format used during SFT.
- Fixed behavior: after chat-template GRPO, completions became compact valid JSON with non-flat rewards and meaningful `final_score` values.
- Limitation: this run forced `central_action` to `{}`, so it trained local phase control but not learned central policy deltas.

### Central-Policy GRPO Preparation

- Status: completed as a partial Kaggle 1B run and uploaded to the Hugging Face Space.
- Schema now allows bounded central policy deltas:
  - `switch_penalty`
  - `queue_urgency_weight`
  - `emergency_boost`
  - `corridor_priority`
  - `balance_penalty`
- Deltas are clipped to `[-0.5, 0.8]` before being sent to the environment.
- GRPO prompts now use chat templates directly, matching the SFT format.
- Central-policy SFT warmup reached strong schema fit (`8/8` validation samples were valid and included `central_action`).
- Partial GRPO run produced 640 episode records before interruption.
- Generated evidence:
  - `plots/central_policy_reward_curve.png`
  - `plots/central_policy_final_score_curve.png`
  - `plots/central_policy_output_quality.png`
  - `results/training_metrics_partial_central_policy.json`
  - `results/training_metrics_partial_central_policy.csv`
- Readout:
  - central policy keys were used consistently,
  - output validity was usually high after parser recovery,
  - reward was non-flat and often positive later in the run,
  - hard-task final score did not yet beat the hard baseline reliably on the 1B run.

### A100 Manual GRPO Central-Policy Run

- Status: completed and uploaded.
- Hardware: Hugging Face A100 JupyterLab Space.
- Run name: `openenv-a100-central-policy-1b-1777183675` (find it in the [W&B project](https://wandb.ai/akshat-arya13-r-v-c-e/traffic-signal-openenv) runs table).
- Reason for stable path: the A100 runtime exposed Unsloth/TRL dtype and optional dependency incompatibilities, so the final run used standard Transformers + PEFT LoRA, manual SFT warmup, and a manual GRPO-style policy optimization loop.
- Episodes recorded: `264`.
- Mean reward: `1.2351`.
- Last-50 mean reward: `1.5061`.
- Best reward: `3.9701`.
- Valid action rate: `99.62%`.
- Central-action rate: `99.62%`.
- Hallucination rate: `0.38%`.
- Mean final score: `0.4750`.
- Best final score: `0.51797`.
- Generated evidence:
  - `plots/a100_central_policy_reward_curve.png`
  - `plots/a100_central_policy_final_score_curve.png`
  - `plots/a100_central_policy_output_quality.png`
  - `results/summary_a100_central_policy.json`
  - `results/training_metrics_a100_central_policy.json`
  - `results/training_metrics_a100_central_policy.csv`
- Adapter uploaded to the Space at `outputs/traffic-lora-a100-central-policy`.

### A100 Final-Score V2 Probe

- Status: stopped and discarded.
- Run name: `openenv-a100-central-policy-v2-finalscore-1777186023`.
- Goal: continue from the A100 adapter with stronger final-score shaping.
- Result: the probe preserved valid JSON and central-action usage, but did not beat the final A100 run's `0.51797` best final score.
- Decision: keep the original A100 run as the final model/artifact.

### Deterministic Policy Sweep Probe

- Status: stopped and not committed as final evidence.
- Goal: replay the best learned action families with a small central-delta grid to see if deterministic policy search could beat the trained model's best final score.
- Result: early focused candidates scored well below the A100 model evidence, so the sweep was stopped to protect submission time.
- Decision: no deterministic sweep result replaces the final A100 trained model.
