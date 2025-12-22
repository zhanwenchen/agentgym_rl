# Model Merger Scripts

These scripts are utilities for merging distributed model checkpoints.

## Scripts

- `model_merger.py` - Merge FSDP checkpoints
- `single_model_merger.py` - Merge single model checkpoints
- `multiple_model_merger.py` - Merge multiple model checkpoints

## Note

These scripts were part of the verl training framework but don't directly depend on verl.
They use PyTorch's distributed tensor functionality to merge checkpoints created during 
distributed training.

Since the training pipeline has been removed, these scripts are kept for reference and 
utility purposes only. They may be useful if you have existing checkpoints to merge.

## Dependencies

These scripts require:
- PyTorch with distributed capabilities
- transformers

Install with:
```bash
pip install torch transformers
```
