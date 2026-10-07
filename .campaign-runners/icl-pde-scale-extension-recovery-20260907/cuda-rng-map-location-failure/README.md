# Archived CUDA RNG resume failure

The partial `K=16, M0=16, seed=1` run was interrupted after training step 5200;
its last authenticated checkpoint is step 5000. On 2026-09-08, the immutable
training source loaded the checkpoint with `map_location=cuda`, which moved the
saved CUDA RNG-state tensor from CPU to CUDA. PyTorch requires a CPU
`torch.uint8` tensor in `torch.cuda.set_rng_state_all`, so resume stopped before
performing any new optimizer update.

The original partial run and automatically published failure marker are kept
here unchanged. Because the immutable campaign manifest forbids patching the
historical training source in place, the selected PDE confirmation restarts
this one seed from step zero in a fresh run directory.
