# One-Liner Dataset Installation

## Quick Install (with dependency workarounds)

**Note**: PDEBench has a dependency conflict (requires torch<1.14, but you have torch>=2.7.1). See workarounds below.

### Option 1: Install compatible packages only
```bash
uv pip install datasets && git clone https://github.com/pdearena/pdearena.git && cd pdearena && uv pip install -e . && cd .. && git clone https://github.com/camlab-ethz/poseidon.git && cd poseidon && uv pip install -e . && cd .. && echo "installed (PDEBench skipped due to torch version conflict)"
```

### Option 2: Manual PDEBench dataset download
Since PDEBench package has dependency issues, you can:
1. Download pre-generated HDF5 datasets directly from: https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986
2. Load them manually in your dataset loader

### Option 3: Use PDEBench from source (bypass package)
```bash
git clone https://github.com/pdebench/PDEBench.git && echo "use PDEBench datasets manually - don't install package due to torch conflict"
```

## What Gets Installed

✅ **datasets** - For Flowbench (Hugging Face)  
✅ **pdearena** - PDEArena benchmark  
✅ **poseidon** - PDEGym datasets  
⚠️ **pdebench** - Skip package, download datasets manually  
⚠️ **pinnacle** - Check for official repository  

## Test Data Information

The small test (`test_small.py`) uses **placeholder/dummy data**:
- Random Gaussian noise (not real PDE data)
- Just tests script structure works
- Results are meaningless but verifies code runs

To test with real data, implement dataset loaders first.
