# Taming Urban Chaos With Hierarchical LLM Control

Urban traffic is a deceptively hard multi-agent coordination problem. A single intersection can be managed with a local rule, but a grid needs longer-horizon reasoning: downstream spillback, corridor formation, emergency routing, phase stability, and fairness across approaches all interact.

Traffic Signal OpenEnv turns that problem into a deterministic LLM training environment. The agent sees a structured text observation with queue lengths, wait times, corridor pressure, incident state, and system-level metrics. It acts through a hierarchical action space: local intersection decisions plus optional central policy deltas that steer global behavior.

## What The Agent Learns

The environment is designed to teach a model something more interesting than one-step action selection. It must coordinate four intersections while avoiding reward hacks like oscillatory switching, all-KEEP collapse, or unsafe priority boosts.

The reward signal is intentionally composable. It combines local efficiency, global coordination, throughput, emergency response, stability, and fairness. This gives the model dense feedback during training while still preserving an episode-level final score for judging whether the policy actually improves traffic flow.

## Training Setup

The training pipeline uses Unsloth and Hugging Face TRL.

We found that direct GRPO was unstable at first: the 1B model often generated prose, malformed JSON, or repeated degenerate actions. The stable recipe became:

1. Supervised fine-tuning warmup to teach the strict traffic-action JSON schema.
2. Schema validation before RL starts.
3. GRPO against the live OpenEnv traffic API, using the same chat-template prompting as SFT.
4. Strict parsing, action sanitization, hallucination penalties, and all-KEEP penalties.
5. W&B monitoring plus saved JSON/CSV metrics and plots.

This is real environment training, not a static dataset. Each GRPO reward call resets or steps the deployed Hugging Face Space environment and records reward, final score, queue metrics, validity, hallucination rate, and policy behavior.

## Results So Far

The central-coordination ablation shows the core environment claim clearly:

- Medium task: about 23% final-score improvement with central coordination enabled.
- Hard multi-task: about 36% final-score improvement with central coordination enabled.

The first stable 1B training runs also showed the practical lesson: schema grounding is mandatory before RL. Once SFT and chat-template GRPO were aligned, completions became compact, valid JSON with non-flat rewards and meaningful score variation.

The next step is central-policy training, where the model learns not only local phase actions but also safe central policy deltas such as `queue_urgency_weight`, `corridor_priority`, `balance_penalty`, `emergency_boost`, and `switch_penalty`.

## Why It Matters

This environment targets scalable oversight: one high-level reasoning model coordinating many local actors under stress. The same pattern appears in fleets, logistics, incident response, robotics, and infrastructure control.

Traffic is familiar enough to understand quickly, but rich enough to expose whether an LLM can actually reason over a changing multi-agent system.

Links:

- Live Space: https://guuru-dev-traffic-signal-openenv-2.hf.space
- Space repository and artifacts: https://huggingface.co/spaces/Guuru-DEV/traffic-signal-openenv-2
- Training notebook: `notebooks/train_colab_FULL.ipynb`
