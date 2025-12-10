# Feature: Add Memory to Agent Training

## Proposal

Add retrieval-based memory: store past experiences and retrieve similar ones as few-shot examples before generation.

```
Memory Bank: [(observation, action, reward), ...]
                    ↓
current_obs → retrieve top-k similar
                    ↓
[retrieved examples] + current_obs → LLM.generate() → action
```

**Why this approach:**
- No architecture changes to actor/critic
- PPO/GRPO work unchanged
- vLLM generation unchanged
- Actual "memory" (episodic recall of past experiences)

## Implementation

1. Add `verl/utils/memory/memory_bank.py`:
   - Store `(obs_embedding, obs_text, action, reward)`
   - FAISS index for fast retrieval
   - Use sentence-transformers for encoding (~22M params)

2. Modify `RolloutHandler.get_generation_prompt()`:
   - Before generation, retrieve k similar past observations
   - Prepend as few-shot examples

3. After each step, store experience in memory bank

## Config

```yaml
memory:
  enabled: true
  k: 3  # retrieve top-3 similar experiences
  min_reward: 0.5  # only store successful experiences
  encoder: "sentence-transformers/all-MiniLM-L6-v2"
```

## Open Questions

1. Store all experiences or only successful ones (reward > threshold)?
2. Task-specific retrieval or cross-task?
