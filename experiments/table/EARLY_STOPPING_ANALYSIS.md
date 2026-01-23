# Analysis: Why Some Frequency Benchmark Runs Stopped at 5k Epochs

## Summary

**Issue**: Some frequency benchmark runs stopped at ~5000 epochs instead of completing the full 10,000 epochs, specifically for `fixWb=True` and low-rank configurations.

## Root Cause

The runs did **NOT** stop due to early stopping logic. Instead, the training processes were **INTERRUPTED/KILLED** after saving checkpoints at 5000 epochs.

### Evidence

1. **Early Stopping Condition**: The code has early stopping at line 253:
   ```python
   if epoch > 300 and avg_loss < 5e-4:
       print(f"early stopping at epoch {epoch} (loss < 5e-4)")
       break
   ```

2. **Actual Behavior**: 
   - Only **1 run** stopped at exactly 5000 epochs: `freq144_48_rank25_fixWbTrue_run27`
   - This run had final loss = `1.633e-02`, which is **> 5e-4**, so early stopping did NOT trigger
   - The config file shows `target_epochs: 3000`, but checkpoint shows `5000 epochs`
   - This suggests the run was **resumed** from a previous 3000-epoch checkpoint and then interrupted

3. **Pattern Analysis**:
   - Most `fixWb=True` + low-rank configs **completed** to 10000 epochs
   - Only 1 run stopped at 5000: `freq144_48_rank25_fixWbTrue_run27`
   - The corresponding `fixWb=False` run completed to 10000 epochs

## Why This Happened

The training process was likely interrupted by one of these reasons:

1. **Process Killed**: The training script was killed (via `pkill`, system restart, or manual termination)
2. **Out of Memory (OOM)**: System ran out of memory and killed the process
3. **Script Crash**: An unhandled exception or error caused the script to exit
4. **Manual Stop**: The user manually stopped the training

## Checkpoint Saving Logic

The code saves checkpoints every 500 epochs (line 238):
```python
if epoch % 500 == 0:
    checkpoint = {...}
    torch.save(checkpoint, checkpoint_path)
```

So when training is interrupted at epoch 5000, a checkpoint is saved, but the training loop never continues to 10000.

## Resume Logic

The code has resume logic (lines 172-202) that should continue from checkpoints:
- If `checkpoint.pth` exists, it resumes from `checkpoint["epoch"] + 1`
- However, if the process was killed, it won't automatically resume unless the script is re-run

## Recommendations

1. **Re-run interrupted configs**: The run `freq144_48_rank25_fixWbTrue_run27` can be resumed by simply re-running the training script - it will automatically load the checkpoint and continue from epoch 5001.

2. **Monitor processes**: Use process monitoring to detect when training is interrupted.

3. **Add error handling**: Wrap the training loop in try-except to catch and log errors before exiting.

4. **Check system resources**: Ensure sufficient memory/GPU resources to prevent OOM kills.

## Affected Configuration

- **Config**: `freq144_48_rank25_fixWbTrue_run27`
- **Target**: 10,000 epochs (originally 3,000, then updated)
- **Actual**: 5,000 epochs
- **Final Loss**: 1.633e-02 (not low enough for early stopping)
- **Status**: INCOMPLETE - needs to be resumed

## Conclusion

The issue is **NOT a bug in the code logic**, but rather **process interruption**. The early stopping condition works correctly (only triggers when loss < 5e-4), and the checkpoint/resume logic is functional. The training simply needs to be resumed for the affected configuration.
