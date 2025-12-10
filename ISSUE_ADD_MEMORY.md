# Feature: Add Memory to Agent Training

## Original Proposal

Use MLP head → embedding → nearest neighbor lookup in a codebook to select actions.

```
prompt → LLM hidden_state → MLP → embedding → NN(codebook) → action
```

## Soundness Issues

The proposal has fundamental problems:

1. **Non-differentiable**: NN lookup breaks gradient flow. PPO/GRPO needs `log_prob` gradients.

2. **Wrong abstraction**: Current system is token-level autoregressive. This replaces it entirely — not "adding memory."

3. **Template args unsolved**: `"pick up {object}"` — how to fill `{object}`?

4. **vLLM incompatible**: vLLM returns generated tokens, not hidden states.

## Simpler Alternative: Retrieval Memory

Keep the existing autoregressive generation. Add memory by retrieving similar past experiences as few-shot examples.

```
Memory Bank: [(observation, action, reward), ...]
                    ↓
current_obs → retrieve top-k similar
                    ↓
[retrieved examples] + current_obs → LLM.generate() → action
```

**Why this works:**
- No architecture changes to actor/critic
- PPO/GRPO work unchanged
- vLLM generation unchanged
- Actual "memory" (episodic recall of past experiences)

## Implementation

1. Add `verl/utils/memory/memory_bank.py`:
   - Store `(obs_embedding, obs_text, action, reward)`
   - FAISS index for fast retrieval
   - Use sentence-transformers for encoding (lightweight, ~22M params)

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
