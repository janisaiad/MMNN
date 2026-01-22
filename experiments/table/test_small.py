"""
we test a small subset of configurations to verify the script works before running on cluster
"""
import sys
from pathlib import Path

# we add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.table.mmnn_vs import (
    AblationConfig, 
    train_one_config, 
    generate_ablation_configs
)
from pathlib import Path
import json


def test_small_subset():
    """we test a small subset of configurations"""
    print("="*80)
    print("RUNNING SMALL TEST SUBSET")
    print("="*80)
    
    # we create test configurations (minimal set)
    test_configs = [
        # we test 1 benchmark, 1 fixWb option, 2 ranks (small and medium)
        AblationConfig(
            benchmark_name="flowbench",
            fixWb=False,
            rank=15,
            num_layers=6,
            hidden_width=1024,
            num_epochs=10,  # we use very few epochs for testing
            batch_size=100,
            n_train_samples=100,  # we use small dataset for testing
            n_test_samples=50,
            log_every=5,  # we log more frequently for testing
            save_every=10,
        ),
        AblationConfig(
            benchmark_name="flowbench",
            fixWb=True,
            rank=15,
            num_layers=6,
            hidden_width=1024,
            num_epochs=10,
            batch_size=100,
            n_train_samples=100,
            n_test_samples=50,
            log_every=5,
            save_every=10,
        ),
        # we test PINN configuration
        AblationConfig(
            benchmark_name="pinnacle",
            fixWb=True,
            rank=15,
            num_layers=6,
            hidden_width=1024,
            num_epochs=10,
            batch_size=100,
            n_train_samples=100,
            n_test_samples=50,
            n_collocation=100,
            n_boundary=20,
            n_initial=20,
            use_pinn=True,
            log_every=5,
            save_every=10,
        ),
    ]
    
    print(f"\ntesting {len(test_configs)} configurations")
    print(f"each with {test_configs[0].num_epochs} epochs (for quick testing)")
    print(f"using {test_configs[0].n_train_samples} training samples\n")
    
    # we set output directory
    base_output_dir = Path("experiments/table/test_results")
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # we run test configurations
    all_results = []
    for idx, config in enumerate(test_configs):
        config.output_dir = base_output_dir / f"test_{config.benchmark_name}_fixWb{config.fixWb}_rank{config.rank}_run{idx}"
        config.output_dir.mkdir(parents=True, exist_ok=True)
        config.device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        
        print(f"\n{'='*80}")
        print(f"TEST {idx+1}/{len(test_configs)}: {config.benchmark_name} | fixWb={config.fixWb} | rank={config.rank}")
        print(f"{'='*80}")
        
        try:
            results = train_one_config(config, config.output_dir)
            all_results.append(results)
            print(f"✓ Test {idx+1} completed successfully")
        except Exception as e:
            print(f"✗ Test {idx+1} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # we save test summary
    summary_path = base_output_dir / "test_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=4)
    
    # we verify outputs
    print(f"\n{'='*80}")
    print("VERIFICATION")
    print(f"{'='*80}")
    
    verification_passed = True
    for idx, config in enumerate(test_configs):
        output_dir = base_output_dir / f"test_{config.benchmark_name}_fixWb{config.fixWb}_rank{config.rank}_run{idx}"
        
        required_files = [
            "config.json",
            "results.json",
            "errors.npz",
            "model_parameters.pth",
            "all_tensors.pt",
            "all_tensors.npz",
        ]
        
        print(f"\nchecking {output_dir.name}:")
        for file in required_files:
            file_path = output_dir / file
            if file_path.exists():
                size = file_path.stat().st_size / 1024  # we get size in KB
                print(f"  ✓ {file} ({size:.1f} KB)")
            else:
                print(f"  ✗ {file} MISSING")
                verification_passed = False
        
        # we check plots (optional)
        plot_files = list(output_dir.glob("*.png"))
        if plot_files:
            print(f"  ✓ {len(plot_files)} plot files")
        else:
            print(f"  ⚠ no plot files (may be normal if save_every > num_epochs)")
    
    print(f"\n{'='*80}")
    if verification_passed:
        print("✓ ALL TESTS PASSED - Script is ready for cluster!")
    else:
        print("✗ SOME TESTS FAILED - Check errors above")
    print(f"{'='*80}")
    print(f"\ntest results saved to: {base_output_dir}")
    print(f"test summary: {summary_path}")
    
    return verification_passed


if __name__ == "__main__":
    success = test_small_subset()
    sys.exit(0 if success else 1)
