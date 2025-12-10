# Feature: Add Memory Module with Codebook-Based Action Embedding

## Summary

Add a memory system that uses a lightweight MLP head on top of the existing LLM encoder to produce action embeddings, which are then matched against a frozen codebook of pre-computed action embeddings via nearest neighbor search.

## Architecture Overview

### Component Breakdown

| Component | Source | Trainable? |
|-----------|--------|------------|
| `policy.encode(prompt)` | **Existing LLM** | Yes (finetuned via GRPO) |
| `policy.encode(action)` → codebook | **Existing LLM** | Frozen at init (or periodically refreshed) |
| `policy.actor_head(hidden)` | **New MLP** | Yes |
| `codebook` | Storage (dict) | N/A |
| `nearest_neighbor()` | Function | N/A |
| `gaussian_log_prob()` | Function | N/A |

### Visual Architecture

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

### Key Design Points

- **Reused**: LLM encoder (bulk of parameters)
- **New**: One small MLP head (~100K params vs ~7B for LLM)
- **No extra model**: No separate inverse dynamics model, no separate embedder

---

## Implementation Plan

### Phase 1: Core Memory Infrastructure

#### 1.1 Create Memory Module (`verl/utils/memory/`)

```
verl/utils/memory/
├── __init__.py
├── codebook.py        # Codebook storage and management
├── actor_head.py      # Small MLP actor head
├── nearest_neighbor.py # NN search utilities
└── memory_config.py   # Configuration dataclass
```

**Files to create:**

- `codebook.py`: Manage action embeddings with methods for:
  - `build_codebook(actions: List[str], encoder: LLM)` - Initialize from action set
  - `refresh_codebook()` - Periodically update embeddings
  - `get_embedding(action_text: str)` - Lookup or compute embedding
  - `save/load` - Persistence utilities

- `actor_head.py`: Small MLP implementation:
  ```python
  class ActorHead(nn.Module):
      def __init__(self, hidden_dim=768, intermediate_dim=256, output_dim=64):
          self.mlp = nn.Sequential(
              nn.Linear(hidden_dim, intermediate_dim),
              nn.ReLU(),
              nn.Linear(intermediate_dim, output_dim)
          )
  ```

- `nearest_neighbor.py`: Efficient NN search:
  - Support for FAISS backend for large codebooks
  - Fallback to torch cosine similarity for small codebooks
  - `gaussian_log_prob()` for probabilistic action selection

#### 1.2 Integration Points in Existing Code

| File | Location | Change |
|------|----------|--------|
| `verl/workers/agent_actor/dp_actor.py` | `~line 58` | Add actor_head after LLM forward, condition on memory |
| `verl/workers/rollout/schemas.py` | `RolloutHandler` | Add memory state to trajectory tracking |
| `verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py` | `~line 252-279` | Compute memory embeddings at each turn |
| `verl/workers/agent_fsdp_workers.py` | `~line 847` | Initialize codebook and actor_head with FSDP |
| `verl/agent_trainer/config/ppo_trainer.yaml` | New section | Add memory configuration options |

---

### Phase 2: Codebook Construction

#### 2.1 Action Space Definition

For each environment, define the action codebook:

```python
# Example for BabyAI
BABYAI_ACTIONS = [
    "turn left", "turn right", "go forward",
    "pick up {object}", "drop {object}",
    "toggle {object}", "done"
]

# Example for WebArena
WEBARENA_ACTIONS = [
    "click [{element}]", "type [{element}] [{text}]",
    "scroll [down]", "scroll [up]",
    "goto [{url}]", "go_back", "go_forward"
]
```

#### 2.2 Codebook Building Strategy

```python
def build_codebook(
    action_templates: List[str],
    encoder: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    use_last_token: bool = True  # vs mean pooling
) -> Dict[str, torch.Tensor]:
    codebook = {}
    for action in action_templates:
        inputs = tokenizer(action, return_tensors="pt")
        with torch.no_grad():
            outputs = encoder(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            if use_last_token:
                emb = hidden[:, -1, :]  # Last token embedding
            else:
                emb = hidden.mean(dim=1)  # Mean pooling
        codebook[action] = emb
    return codebook
```

---

### Phase 3: Training Integration

#### 3.1 Modified Forward Pass

```python
# In dp_actor.py
class DataParallelPPOActor:
    def __init__(self, ...):
        self.actor_head = ActorHead(
            hidden_dim=self.config.hidden_size,
            output_dim=self.config.memory_output_dim
        )
        self.codebook = Codebook(...)

    def forward(self, input_ids, attention_mask, ...):
        # Existing: Get LLM hidden states
        outputs = self.model(input_ids, attention_mask, ...)
        hidden_states = outputs.hidden_states[-1]

        # NEW: Project to action embedding space
        action_emb = self.actor_head(hidden_states[:, -1, :])

        # NEW: Nearest neighbor in codebook
        nearest_action, distance = self.codebook.nearest_neighbor(action_emb)

        # NEW: Compute gaussian log prob for RL
        log_prob = gaussian_log_prob(action_emb, nearest_action, sigma=self.sigma)

        return log_prob, nearest_action
```

#### 3.2 Loss Function Updates

```python
# Add embedding alignment loss (optional regularization)
def compute_actor_loss(self, ...):
    # Existing PPO loss
    ppo_loss = ...

    # NEW: Optional embedding regularization
    if self.config.use_embedding_loss:
        emb_loss = F.mse_loss(action_emb, target_action_emb)
        total_loss = ppo_loss + self.config.emb_loss_weight * emb_loss
    else:
        total_loss = ppo_loss

    return total_loss
```

---

### Phase 4: Configuration

#### 4.1 New Config Section in `ppo_trainer.yaml`

```yaml
memory:
  enabled: true

  codebook:
    build_strategy: "from_actions"  # or "from_trajectories"
    refresh_interval: 1000  # steps between codebook refresh (0 = frozen)
    embedding_dim: 64
    use_last_token: true

  actor_head:
    hidden_dim: 768  # Should match LLM hidden size
    intermediate_dim: 256
    output_dim: 64
    dropout: 0.1

  nearest_neighbor:
    backend: "faiss"  # or "torch"
    metric: "cosine"  # or "l2"

  training:
    sigma: 1.0  # For gaussian log prob
    embedding_loss_weight: 0.0  # Set > 0 for regularization
```

---

### Phase 5: Testing and Validation

#### 5.1 Unit Tests

```
tests/memory/
├── test_codebook.py
├── test_actor_head.py
├── test_nearest_neighbor.py
└── test_integration.py
```

#### 5.2 Integration Tests

- Verify codebook builds correctly from action templates
- Verify actor_head gradients flow correctly
- Verify nearest neighbor retrieval accuracy
- End-to-end training loop with memory enabled

#### 5.3 Benchmarks

- Memory overhead comparison
- Training speed comparison (should be minimal impact)
- Action selection accuracy vs baseline

---

## Migration Path

### Backward Compatibility

- Memory module is **opt-in** via `memory.enabled: true`
- Existing training configs work unchanged
- Checkpoints remain compatible (actor_head saved separately)

### Gradual Rollout

1. **Week 1**: Implement core memory infrastructure
2. **Week 2**: Integrate with actor forward pass
3. **Week 3**: Add codebook building for each environment
4. **Week 4**: Testing and benchmarking
5. **Week 5**: Documentation and examples

---

## Open Questions

1. **Codebook refresh strategy**: Should we refresh periodically during training, or keep frozen?
2. **Action templates vs learned**: Should codebook actions be hand-crafted templates or learned from trajectories?
3. **Embedding dimension**: What's the optimal output dimension for the actor head? (64, 128, 256?)
4. **Multi-turn memory**: Should we maintain episodic memory across turns, or just use the codebook for action selection?

---

## References

- Current actor implementation: `verl/workers/agent_actor/dp_actor.py`
- Current critic implementation: `verl/workers/agent_critic/dp_critic.py`
- Rollout handling: `verl/workers/rollout/schemas.py`
- Training config: `verl/agent_trainer/config/ppo_trainer.yaml`
