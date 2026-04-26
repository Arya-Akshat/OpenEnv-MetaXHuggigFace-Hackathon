# Training Run Log

This file tracks real training attempts and generated evidence for the hackathon submission.

## Current Evidence

- Environment: Hugging Face Space `Guuru-DEV/traffic-signal-openenv-2`
- Model family: `unsloth/Llama-3.2-1B-Instruct`
- Training stack: Unsloth + Hugging Face TRL
- Monitoring: Weights & Biases project `traffic-signal-openenv`
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

- Status: configured, ready for next Kaggle run.
- Schema now allows bounded central policy deltas:
  - `switch_penalty`
  - `queue_urgency_weight`
  - `emergency_boost`
  - `corridor_priority`
  - `balance_penalty`
- Deltas are clipped to `[-0.5, 0.8]` before being sent to the environment.
- GRPO prompts now use chat templates directly, matching the SFT format.

### Planned A100 Run

- Status: pending.
- Goal: after central-policy behavior is stable on the 1B model, run the same pipeline on a larger model using an A100-backed Hugging Face Job.
- Success criteria:
  - high schema-valid completion rate,
  - non-flat reward distribution,
  - improved hard-task `final_score`,
  - readable plots saved in `plots/`,
  - metrics saved in `results/`,
  - adapter uploaded to Hugging Face Hub/Space, not committed to Git.
