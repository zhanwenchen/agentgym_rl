# Feature: Add Memory Module with Codebook-Based Action Embedding

## Summary

Add a memory system that uses a lightweight MLP head on top of the existing LLM encoder to produce action embeddings, which are then matched against a frozen codebook of pre-computed action embeddings via nearest neighbor search.

---

## CRITICAL: Soundness Analysis

After reviewing the current architecture (`dp_actor.py`, `core_algos.py`), there are several **fundamental issues** with the proposed design that need to be addressed before implementation.

### Issue 1: Fundamental Architecture Mismatch

**Current system** (token-level autoregressive):
```
input_ids → LLM → logits[vocab_size] → sample token → repeat
log_prob = Σ log P(token_i | context)
```

**Proposed system** (embedding-based):
```
prompt → LLM hidden_state → MLP → embedding → NN lookup → action string
```

These are **incompatible paradigms**. The proposal replaces autoregressive generation with a completely different action selection mechanism. This isn't "adding memory" — it's replacing the entire action head.

### Issue 2: Non-Differentiable Nearest Neighbor

The NN lookup `argmin(||action_emb - codebook||)` is **non-differentiable**. Gradients cannot flow through discrete selection.

The `gaussian_log_prob()` workaround:
```python
log_prob = -||action_emb - nearest_codebook_entry||² / (2σ²)
```

Has problems:
- Gradient pushes `action_emb` toward the **nearest** entry, not the **correct** one
- In RL, we don't have supervised labels — only rewards
- High-reward "wrong" actions will attract the embedding regardless

### Issue 3: GRPO/PPO Training Dynamics Break

Current PPO in `dp_actor.py:242-248`:
```python
entropy, log_prob = self._forward_micro_batch(...)  # token-level
pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
    old_log_prob=old_log_prob,
    log_prob=log_prob,  # shape: (bs, response_length)
    advantages=advantages,
    ...
)
```

The policy loss expects **per-token log probabilities** over the response. With embedding-based selection:
- Only **one** log_prob per action (not per token)
- No entropy over vocabulary (embedding is deterministic given hidden state)
- `old_log_prob` vs `log_prob` ratio is ill-defined (what's the "old" embedding?)

### Issue 4: Template Arguments (`{object}`, `{element}`)

The codebook has templates like `"pick up {object}"`. How to fill arguments?
- **Option A**: Enumerate all objects → codebook explosion (1000s of entries)
- **Option B**: Separate model for arguments → adds complexity, defeats "no extra model"
- **Option C**: LLM generates arguments → back to autoregressive for part of action

### Issue 5: vLLM Rollout Integration

Current rollout (`vllm_rollout.py`) uses vLLM for **generation**:
```python
outputs = self.inference_engine.generate(...)  # Returns token sequences
```

The proposed system needs **hidden states**, not generated tokens. vLLM's inference API doesn't expose intermediate hidden states efficiently.

### Issue 6: Multi-Turn Context

Current multi-turn concatenates everything:
```
[prompt][action1][obs1][action2][obs2]...
```

With embedding-based selection:
- Re-encode entire history each turn? (expensive, O(n²) over turns)
- Or maintain a recurrent state? (not in current LLM architecture)

---

## Recommended Alternatives

### Alternative A: Keep Autoregressive + Add Retrieval Memory (Recommended)

If the goal is **memory**, keep generation intact and add retrieval:

```
┌─────────────────────────────────────────────────────────┐
│                   RETRIEVAL MEMORY                       │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Memory Bank: [(obs, action, reward), ...]       │   │
│  │  Query: current_obs → retrieve top-k similar     │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    EXISTING LLM                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │  [retrieved_examples] + prompt ──► LLM ──► action│   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

**Pros**:
- Preserves differentiability
- Works with existing PPO/GRPO
- Actual "memory" (episodic recall)
- Compatible with vLLM

**Implementation**:
- Add memory bank to `RolloutHandler`
- Retrieve similar past experiences before generation
- Prepend to prompt as few-shot examples

### Alternative B: Action Proposal + Reranking

Hybrid that keeps generation but adds embedding-based value estimation:

```
┌─────────────────────────────────────────────────────────┐
│              STEP 1: Generate N Candidates              │
│  ┌──────────────────────────────────────────────────┐   │
│  │  prompt ──► LLM.generate(n=5) ──► [a1, a2, ...]  │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              STEP 2: Embed & Score                      │
│  ┌──────────────────────────────────────────────────┐   │
│  │  [a1, a2, ...] ──► Encoder ──► [e1, e2, ...]     │   │
│  │  Value head: ei ──► v_i (scalar)                 │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
                   Select: argmax(v_i)
```

**Pros**:
- LLM generates valid actions (including arguments)
- Embedding space for value estimation
- Differentiable (train value head with TD/MC returns)

### Alternative C: VQ-VAE Style (Major Refactor)

If you truly want embedding-based discrete actions:

1. **Use Gumbel-Softmax** for differentiable codebook selection
2. **Pre-train codebook** via behavioral cloning on expert trajectories
3. **Use continuous RL** (SAC/TD3) in embedding space
4. **Decoder** to convert selected codebook entry back to action string

This is a **major architectural change**, not incremental.

---

## Revised Architecture (Alternative A)

### Component Breakdown

| Component | Source | Trainable? |
|-----------|--------|------------|
| `memory_bank` | Storage | N/A |
| `retriever.encode(obs)` | **Existing LLM** or sentence-transformers | Frozen |
| `retriever.search(query, k)` | FAISS/torch | N/A |
| `LLM.generate(prompt + retrieved)` | **Existing LLM** | Yes (GRPO) |

### Visual Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MEMORY BANK                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Entry 1: (obs_emb, obs_text, action, reward)    │   │
│  │  Entry 2: (obs_emb, obs_text, action, reward)    │   │
│  │  ...                                              │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                           │
        current_obs ──► encode ──► query_emb
                           │
                           ▼
              FAISS.search(query_emb, k=3)
                           │
                           ▼
         Retrieved: [(obs1, action1), (obs2, action2), ...]
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    EXISTING LLM                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │  [Retrieved examples as few-shot]                │   │
│  │  [Current observation]                           │   │
│  │  ──► LLM.generate() ──► action                   │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## Implementation Plan (Revised for Alternative A)

### Phase 1: Memory Infrastructure

#### 1.1 Create Memory Module (`verl/utils/memory/`)

```
verl/utils/memory/
├── __init__.py
├── memory_bank.py     # Experience storage
├── retriever.py       # Embedding + search
└── memory_config.py   # Configuration
```

**memory_bank.py**:
```python
@dataclass
class MemoryEntry:
    observation: str
    observation_emb: torch.Tensor  # Pre-computed
    action: str
    reward: float
    task_id: str  # For task-specific retrieval

class MemoryBank:
    def __init__(self, max_size: int = 100000):
        self.entries: List[MemoryEntry] = []
        self.index: faiss.Index = None  # Built lazily

    def add(self, obs: str, obs_emb: torch.Tensor, action: str, reward: float, task_id: str):
        ...

    def search(self, query_emb: torch.Tensor, k: int = 3, task_id: str = None) -> List[MemoryEntry]:
        ...

    def build_index(self):
        """Build FAISS index from observation embeddings"""
        ...
```

**retriever.py**:
```python
class Retriever:
    def __init__(self, encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Use lightweight encoder, NOT the 7B LLM
        self.encoder = SentenceTransformer(encoder_name)

    def encode(self, text: str) -> torch.Tensor:
        return self.encoder.encode(text, convert_to_tensor=True)
```

### Phase 2: Integration with Rollout

#### 2.1 Modify RolloutHandler (`schemas.py`)

```python
class RolloutHandler:
    def __init__(self, ..., memory_bank: MemoryBank = None, retriever: Retriever = None):
        self.memory_bank = memory_bank
        self.retriever = retriever

    def get_generation_prompt(self, include_memory: bool = True) -> str:
        prompt = self._build_base_prompt()

        if include_memory and self.memory_bank:
            # Retrieve similar past experiences
            query_emb = self.retriever.encode(self.current_observation)
            memories = self.memory_bank.search(query_emb, k=3)

            # Format as few-shot examples
            memory_text = self._format_memories(memories)
            prompt = memory_text + "\n\n" + prompt

        return prompt

    def store_experience(self, obs: str, action: str, reward: float):
        """Called after each step to populate memory"""
        if self.memory_bank:
            obs_emb = self.retriever.encode(obs)
            self.memory_bank.add(obs, obs_emb, action, reward, self.task_id)
```

#### 2.2 Modify vLLM Rollout (`vllm_rollout.py`)

```python
# In generate_sequences, after getting action and reward:
if self.memory_enabled:
    handler.store_experience(
        obs=current_observation,
        action=generated_action,
        reward=step_reward
    )
```

### Phase 3: Configuration

```yaml
memory:
  enabled: true

  bank:
    max_size: 100000
    persist_path: "checkpoints/memory_bank.pt"

  retriever:
    encoder: "sentence-transformers/all-MiniLM-L6-v2"  # 22M params, fast
    # OR use LLM hidden states (slower but consistent):
    # encoder: "llm_hidden_state"

  retrieval:
    k: 3  # Number of examples to retrieve
    task_specific: true  # Only retrieve from same task type
    min_reward: 0.5  # Only retrieve successful experiences

  prompt_format: |
    Here are similar past experiences:
    {memories}

    Now handle the current situation:
```

---

## Comparison: Original vs Revised

| Aspect | Original (Codebook) | Revised (Retrieval) |
|--------|---------------------|---------------------|
| Differentiability | Broken (NN lookup) | Preserved (standard generation) |
| PPO/GRPO compatibility | Requires major changes | Works out of the box |
| vLLM integration | Needs hidden states | Uses standard generation |
| Action arguments | Unsolved | Handled by LLM |
| True "memory" | No (static codebook) | Yes (episodic retrieval) |
| New parameters | ~100K (MLP head) | ~22M (sentence encoder) |

---

## Open Questions

1. **Retrieval encoder**: Use lightweight sentence-transformers (fast, 22M) or LLM hidden states (slower, consistent)?
2. **Memory population**: Only store successful trajectories (reward > threshold) or all experiences?
3. **Cross-task transfer**: Allow retrieval across different task types, or keep task-specific?
4. **Memory refresh**: Prune low-reward experiences over time?

---

## Original Proposal (Preserved for Reference)

<details>
<summary>Click to expand original codebook-based proposal</summary>

### Original Architecture

| Component | Source | Trainable? |
|-----------|--------|------------|
| `policy.encode(prompt)` | **Existing LLM** | Yes (finetuned via GRPO) |
| `policy.encode(action)` → codebook | **Existing LLM** | Frozen at init (or periodically refreshed) |
| `policy.actor_head(hidden)` | **New MLP** | Yes |
| `codebook` | Storage (dict) | N/A |
| `nearest_neighbor()` | Function | N/A |
| `gaussian_log_prob()` | Function | N/A |

```
┌─────────────────────────────────────────────────────────┐
│                    EXISTING LLM                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │  prompt ──► LLM Encoder ──► hidden_state         │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    NEW: Small MLP                       │
│  ┌──────────────────────────────────────────────────┐   │
│  │  hidden_state ──► actor_head ──► action_emb      │   │
│  │                   (e.g., 768 → 256 → 64)         │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│               CODEBOOK (from LLM, frozen)               │
│  ┌──────────────────────────────────────────────────┐   │
│  │  "turn on X" ──► LLM.encode() ──► emb_1          │   │
│  │  "pick up X" ──► LLM.encode() ──► emb_2          │   │
│  │  ...                                              │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
              nearest_neighbor(action_emb, codebook)
                           │
                           ▼
                     "turn on lamp"
```

</details>

---

## References

- Current actor implementation: `verl/workers/agent_actor/dp_actor.py`
- Current critic implementation: `verl/workers/agent_critic/dp_critic.py`
- Rollout handling: `verl/workers/rollout/schemas.py`
- Training config: `verl/agent_trainer/config/ppo_trainer.yaml`
- PPO loss computation: `verl/agent_trainer/ppo/core_algos.py:276-307`
