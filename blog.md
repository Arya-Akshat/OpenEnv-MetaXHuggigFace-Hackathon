# Taming Urban Chaos With a Tiny LLM

Traffic looks simple until four intersections start arguing with each other. One green light clears a queue, but it can also flood the next junction. A local rule can be smart for one corner and still create gridlock for the whole block.

Traffic Signal OpenEnv turns that problem into a deterministic LLM environment. The model reads a compact text observation with queues, waits, corridor pressure, incidents, and system metrics. Then it returns one JSON action: local phase choices for `NW`, `NE`, `SW`, and `SE`, plus small central-policy deltas such as `queue_urgency_weight` and `corridor_priority`.

## The Trick

We did not chase a huge model. We used a 1B instruct model and treated training like systems debugging.

The first RL attempts failed in very LLM ways: prose instead of JSON, malformed keys, repeated `KEEP`, and reward-looking behavior that did not always improve traffic. The stable recipe was:

1. Teach the JSON schema first with supervised fine-tuning.
2. Gate RL on schema validation.
3. Penalize hallucinations, all-KEEP collapse, and invalid actions.
4. Let the model tune both local actions and central policy deltas.
5. Track every run with W&B, CSV/JSON metrics, plots, and a human run log.

On A100, the fastest Unsloth/TRL path hit runtime dtype and dependency issues, so the final run used standard Transformers + PEFT LoRA with a manual GRPO-style loop. That was the right trade: smaller model, cleaner control, better evidence.

## What We Got

The final A100 run recorded 264 environment episodes:

- `99.62%` valid JSON actions
- `99.62%` central-action usage
- `0.38%` hallucination rate
- `1.506` last-50 mean reward
- `0.51797` best hard-task final score

The broader environment ablation also supports the central-control idea: central coordination improved medium-task final score by about `23%` and hard multi-task score by about `36.2%`.

## Why It Matters

This is a small traffic world, but the shape is familiar: one high-level reasoner coordinating many local actors under stress. The same pattern shows up in fleets, warehouses, incident response, robotics, and infrastructure control.

The fun part is that the model did not win by talking more. It won by learning to speak less: one compact JSON object, sent at the right time, with the right central nudge.

Links:

- Weights & Biases (training runs and metrics): https://wandb.ai/akshat-arya13-r-v-c-e/traffic-signal-openenv
- Live environment Space: https://guuru-dev-traffic-signal-openenv-2.hf.space
- Space repository and artifacts: https://huggingface.co/spaces/Guuru-DEV/traffic-signal-openenv-2
- GitHub repository: [OpenEnv-MetaXHuggingFace-Hackathon](https://github.com/Arya-Akshat/OpenEnv-MetaXHuggigFace-Hackathon)
- Training notebook: `notebooks/train_colab_FULL.ipynb`
- Run log: `results/run_log.md`
