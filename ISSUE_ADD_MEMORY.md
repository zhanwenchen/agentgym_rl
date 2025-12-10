# Add Retrieval-Based Memory for Agent Training

## Motivation

Currently, agents have no mechanism to recall past experiences during rollout. Each episode starts fresh with no memory of what worked before in similar situations.

## Proposed Solution

Add a memory bank that stores past `(observation, action, reward)` tuples. Before generating an action, retrieve similar past experiences and prepend them as few-shot examples.

```
current_obs → retrieve similar experiences → prepend to prompt → generate action
```

## Benefits

- Enables learning from past successes without architecture changes
- Compatible with existing PPO/GRPO training
- Lightweight: uses small encoder (~22M params) separate from main LLM

## Implementation Sketch

1. New `MemoryBank` class with FAISS index for retrieval
2. Hook into `RolloutHandler.get_generation_prompt()` to prepend retrieved examples
3. Store experiences after each environment step

## Open Questions

- Store all experiences or only successful ones?
- Task-specific vs cross-task retrieval?
