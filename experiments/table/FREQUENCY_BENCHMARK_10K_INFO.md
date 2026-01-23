# Frequency Benchmark - 10K Epochs (Resume from Existing Models)

## Updates

**Epochs**: Changed from 3000 → **10000**

## Checkpoint System

The script now supports resuming from existing training:

1. **Checkpoint Loading**: If `checkpoint.pth` exists, loads:
   - Model state
   - Optimizer state
   - Scheduler state
   - Training history (losses, errors)
   - Resumes from saved epoch

2. **Model Loading**: If `model_parameters.pth` exists (from previous 3k epoch training):
   - Loads model weights
   - Loads training history from `results.json`
   - Resumes from epoch 3001 (continues to 10k)

3. **Fresh Start**: If no checkpoint/model exists, starts fresh

## Checkpoint Saving

- **Every 500 epochs**: Saves checkpoint with full state
- **Final checkpoint**: Saved at end of training
- **Model file**: `model_parameters.pth` saved at end

## Current Status

**Process**: Running in background
**Log**: `test_frequency_benchmark_10k.log`
**Results**: `experiments/table/results_frequency_benchmark/`

## Note

Existing results from 3k epoch training don't have saved models, so they will:
- Start fresh training to 10k epochs
- Save checkpoints during training for future resumes

Future runs will be able to resume from checkpoints!
