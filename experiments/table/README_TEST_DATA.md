# Test Data Information

## Small Test (`test_small.py`) Uses Placeholder Data

The small test script (`experiments/table/test_small.py`) currently uses **placeholder/dummy data**, not real datasets.

### What Data is Used

1. **Flowbench tests** (2 configurations):
   - Uses `PlaceholderPDEDataset` 
   - Generates random Gaussian data: `np.random.randn(n_samples, input_dim)`
   - **Not real flow simulation data**
   - Just tests that the script structure works

2. **PINNacle test** (1 configuration):
   - Uses `PlaceholderPINNDataset`
   - Generates random data for:
     - Data points: `np.random.randn(n_samples, input_dim)`
     - Collocation points: `np.random.randn(n_collocation, input_dim)`
     - Boundary points: `np.random.randn(n_boundary, input_dim)`
     - Initial points: `np.random.randn(n_initial, input_dim)`
   - **Not real PDE data**
   - Just tests that PINN loss computation structure works

### Purpose of Test

The test verifies:
- ✅ Script structure works
- ✅ Model training runs
- ✅ Saving mechanisms work
- ✅ File outputs are created correctly
- ✅ No syntax/runtime errors

**It does NOT verify:**
- ❌ Actual dataset loading
- ❌ Real PDE solving performance
- ❌ Meaningful results

### To Test with Real Data

After installing datasets, you would need to:
1. Update `load_pde_dataset()` to load real data
2. Run the test again to verify real data loading works
3. Then run the full ablation study

### Current Test Configuration

- **Epochs**: 10 (very few, just for testing)
- **Training samples**: 100 (small)
- **Test samples**: 50 (small)
- **Data**: Random Gaussian noise (placeholder)

This is intentionally minimal to test quickly (~1-2 seconds per config).
