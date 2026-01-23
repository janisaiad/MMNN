#!/usr/bin/env python3
"""
we test MMNN with different frequencies and fixWb/rank combinations
based on the working config from benchmark.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import os
import json
from pathlib import Path
from datetime import datetime
import sys

# we add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.table.mmnn_vs import MMNN


def generate_frequency_configs():
    """we generate configurations with different frequencies"""
    configs = []
    
    # we define frequency pairs: (freq1, freq2) for cos(freq1*pi*x^2) - 0.8*cos(freq2*pi*x^2)
    frequency_pairs = [
        (36, 12),   # base frequency
        (72, 24),   # 2x frequency
        (144, 48),  # 4x frequency (scaled from 12/24/48 pattern)
    ]
    
    # we define ranks to test
    ranks = [10, 15, 20, 25, 50]
    
    # we define fixWb options
    fixWb_options = [False, True]
    
    # we base config
    base_config = {
        "num_layers": 8,
        "hidden_width": 777,
        "input_rank": 1,
        "output_rank": 1,
        "use_resnet": False,
        "num_epochs": 3000,
        "batch_size": 100,
        "lr_init": 0.001,
        "lr_gamma": 0.9,
        "lr_step_size": 100,
        "interval": [-1, 1],
        "show_plot": False,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "dtype": "torch.float32"
    }
    
    # we generate all combinations
    for freq1, freq2 in frequency_pairs:
        # we scale training samples with frequency (higher freq = more samples needed)
        # base: 1000 samples for freq (36,12)
        # scale factor: max(freq1, freq2) / 36
        scale_factor = max(freq1, freq2) / 36.0
        num_training_samples = int(1000 * scale_factor)
        num_test_samples = int(1234 * scale_factor)  # we keep same ratio
        
        for rank in ranks:
            for fixWb in fixWb_options:
                config = base_config.copy()
                config.update({
                    "hidden_rank": rank,
                    "num_training_samples": num_training_samples,
                    "num_test_samples": num_test_samples,
                    "function": f"cos({freq1}*pi*x^2) - 0.8*cos({freq2}*pi*x^2)",
                    "freq1": freq1,
                    "freq2": freq2,
                    "fixWb": fixWb,
                })
                configs.append(config)
    
    return configs


def func_from_string(func_str, x):
    """we evaluate function from string"""
    # we parse frequencies from string like "cos(36*pi*x^2) - 0.8*cos(12*pi*x^2)"
    import re
    match = re.match(r'cos\((\d+)\*pi\*x\^2\) - 0\.8\*cos\((\d+)\*pi\*x\^2\)', func_str)
    if match:
        freq1, freq2 = int(match.group(1)), int(match.group(2))
        y = np.cos(freq1 * np.pi * x**2) - 0.8 * np.cos(freq2 * np.pi * x**2)
        return y
    else:
        raise ValueError(f"cannot parse function: {func_str}")


def train_one_config(config, output_dir):
    """we train one configuration"""
    device = torch.device(config["device"])
    mydtype = torch.get_default_dtype()
    
    print(f"\n{'='*80}")
    print(f"Training: freq=({config['freq1']},{config['freq2']}), rank={config['hidden_rank']}, fixWb={config['fixWb']}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}")
    
    # we set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # we build model
    ranks = [config["input_rank"]] + [config["hidden_rank"]] * config["num_layers"] + [config["output_rank"]]
    widths = [config["hidden_width"]] * (config["num_layers"] + 1)
    
    model = MMNN(
        ranks=ranks,
        widths=widths,
        device=device,
        ResNet=config["use_resnet"],
        fixWb=config["fixWb"]
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total parameters: {total_params:,}")
    print(f"trainable parameters: {trainable_params:,}")
    
    # we create function
    def func(x):
        return func_from_string(config["function"], x)
    
    # we create datasets
    x_train = np.linspace(*config["interval"], config["num_training_samples"]).reshape([-1, 1])
    y_train = func(x_train)
    x_train = torch.tensor(x_train, device=device, dtype=mydtype)
    y_train = torch.tensor(y_train, device=device, dtype=mydtype)
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    
    # we create test data
    x_test = np.random.rand(config["num_test_samples"]) * 2 - 1
    y_test = func(x_test)
    x_test_tensor = torch.tensor(x_test.reshape([-1, 1]), device=device, dtype=mydtype)
    y_test_tensor = torch.tensor(y_test.reshape([-1, 1]), device=device, dtype=mydtype)
    
    # we setup training
    optimizer = optim.Adam(model.parameters(), lr=config["lr_init"])
    scheduler = StepLR(optimizer, step_size=config["lr_step_size"], gamma=config["lr_gamma"])
    criterion = nn.MSELoss()
    
    # we train
    all_losses = []
    errors_train = []
    errors_test = []
    errors_test_max = []
    start_time = time.time()
    
    for epoch in range(1, config["num_epochs"] + 1):
        epoch_losses = []
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        all_losses.append(avg_loss)
        scheduler.step()
        
        # we evaluate on test set
        if epoch % 50 == 0 or epoch == 1:
            with torch.no_grad():
                y_pred = model(x_test_tensor)
                test_error = criterion(y_pred, y_test_tensor).item()
                test_error_max = torch.max(torch.abs(y_pred - y_test_tensor)).item()
                
                errors_train.append(avg_loss)
                errors_test.append(test_error)
                errors_test_max.append(test_error_max)
                
                print(f"Epoch {epoch}/{config['num_epochs']}: "
                      f"train={avg_loss:.4e}, test={test_error:.4e}, test_max={test_error_max:.4e}")
        
        # we early stop if loss is very low
        if epoch > 300 and avg_loss < 5e-4:
            print(f"early stopping at epoch {epoch} (loss < 5e-4)")
            break
    
    training_time = time.time() - start_time
    
    # we save results
    results = {
        "config": config,
        "final_train_error": float(errors_train[-1]) if errors_train else None,
        "final_test_error": float(errors_test[-1]) if errors_test else None,
        "final_test_error_max": float(errors_test_max[-1]) if errors_test_max else None,
        "training_time_seconds": float(training_time),
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "epochs_run": int(len(all_losses)),
        "all_losses": [float(l) for l in all_losses],
        "errors_train": [float(e) for e in errors_train],
        "errors_test": [float(e) for e in errors_test],
        "errors_test_max": [float(e) for e in errors_test_max],
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    # we save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    # we plot final prediction
    x_plot = np.linspace(*config["interval"], 1000)
    x_plot_tensor = torch.tensor(x_plot.reshape([-1, 1]), device=device, dtype=mydtype)
    with torch.no_grad():
        y_plot_nn = model(x_plot_tensor).cpu().numpy().reshape([-1])
    y_plot_true = func(x_plot)
    
    fig = plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_plot_true, 'b-', label='True function', linewidth=2)
    plt.plot(x_plot, y_plot_nn, 'r--', label='Learned network', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    config_str = f"freq=({config['freq1']},{config['freq2']}), rank={config['hidden_rank']}, fixWb={config['fixWb']}"
    plt.title(f'Final Prediction\n{config_str}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'final_prediction.png', dpi=100)
    plt.close()
    
    # we plot loss evolution
    fig = plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, len(all_losses)+1), all_losses, 'b-', linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title(f'Training Loss Evolution\n{config_str}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_evolution.png', dpi=100)
    plt.close()
    
    # we plot error evolution
    if errors_test:
        fig = plt.figure(figsize=(10, 6))
        epochs_logged = np.linspace(1, len(all_losses), len(errors_train)) * 50
        plt.plot(epochs_logged, np.log10(errors_train), label="log10 train error", linewidth=2)
        plt.plot(epochs_logged, np.log10(errors_test), label="log10 test error", linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('log10(error)')
        plt.title(f'Error Evolution\n{config_str}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'error_evolution.png', dpi=100)
        plt.close()
    
    print(f"✓ completed: {output_dir.name}")
    return results


def main():
    """we run frequency benchmark"""
    print("="*80)
    print("Frequency Benchmark: Testing MMNN with different frequencies")
    print("="*80)
    
    configs = generate_frequency_configs()
    print(f"\ngenerated {len(configs)} configurations")
    print(f"  frequency pairs: (36,12), (72,24), (144,48)")
    print(f"  ranks: {[10, 15, 20, 25, 50]}")
    print(f"  fixWb: False, True")
    
    # we create output directory
    base_output_dir = Path("experiments/table/results_frequency_benchmark")
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # we setup logging
    log_file = base_output_dir / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    class Tee:
        def __init__(self, file_path):
            self.file = open(file_path, 'w')
            self.stdout = sys.stdout
        
        def write(self, text):
            self.file.write(text)
            self.file.flush()
            self.stdout.write(text)
        
        def flush(self):
            self.file.flush()
            self.stdout.flush()
    
    tee = Tee(log_file)
    sys.stdout = tee
    sys.stderr = tee
    
    try:
        print(f"\nstarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"log file: {log_file}")
        print(f"output directory: {base_output_dir}")
        
        all_results = []
        start_time = time.time()
        
        for idx, config in enumerate(configs):
            # we create output directory for this config
            config_name = (f"freq{config['freq1']}_{config['freq2']}_"
                          f"rank{config['hidden_rank']}_"
                          f"fixWb{config['fixWb']}_run{idx}")
            output_dir = base_output_dir / config_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            elapsed = time.time() - start_time
            remaining = len(configs) - idx
            avg_time = elapsed / (idx + 1) if idx > 0 else 0
            estimated_remaining = avg_time * remaining
            
            print(f"\n{'='*80}")
            print(f"CONFIG {idx+1}/{len(configs)}")
            print(f"  freq: ({config['freq1']}, {config['freq2']})")
            print(f"  rank: {config['hidden_rank']}")
            print(f"  fixWb: {config['fixWb']}")
            print(f"  elapsed: {elapsed/3600:.2f} hours")
            print(f"  estimated remaining: {estimated_remaining/3600:.2f} hours")
            print(f"{'='*80}")
            
            try:
                results = train_one_config(config, output_dir)
                all_results.append(results)
            except Exception as e:
                print(f"✗ Config {idx+1} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        total_time = time.time() - start_time
        
        # we save summary
        summary_path = base_output_dir / "frequency_benchmark_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=4)
        
        print(f"\n{'='*80}")
        print("TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"completed: {len(all_results)}/{len(configs)} configurations")
        print(f"total time: {total_time/3600:.2f} hours")
        print(f"results saved to: {base_output_dir}")
        print(f"summary saved to: {summary_path}")
        print(f"completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
    finally:
        sys.stdout = tee.stdout
        sys.stderr = tee.stdout
        tee.file.close()


if __name__ == "__main__":
    main()
