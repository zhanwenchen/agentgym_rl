# AgentGym-RL Utilities

This directory contains standalone utilities extracted from the verl framework.

## Memory Bank (`memory/`)

A FAISS-based memory bank for storing and retrieving agent experiences.

### Dependencies

The memory bank requires additional dependencies not included in the minimal setup:
```bash
pip install faiss-cpu torch sentence-transformers
```

### Usage

See `test_memory_bank.py` in the root directory for example usage.

### Note

This is a standalone utility that can function independently of the full training pipeline.
It was extracted during the pipeline minimization to preserve useful functionality.
