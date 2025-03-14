#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmark script for multi-GPU training with PyTorch Lightning.

This script benchmarks training performance across different:
- GPU configurations (1, 2, 4, 8 GPUs)
- Training strategies (DP, DDP)
- Batch sizes (64, 128, 256, 512)

Metrics collected:
- Training time per epoch
- Memory usage
- Throughput (samples/second)
- GPU utilization
"""

import os
import time
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pytorch_lightning.callbacks import Callback


# Define a simple model for benchmarking
class BenchmarkModel(pl.LightningModule):
    def __init__(self, hidden_dim=512, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        
        # Model architecture - simple MLP for MNIST
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)
        )
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


# Callback to measure GPU memory usage and timing
class BenchmarkCallback(Callback):
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.end_time = None
        self.epoch_times = []
        self.max_memory = 0
        self.batch_times = []
        self.batch_start_time = None
        
    def on_train_epoch_start(self, trainer, pl_module):
        self.start_time = time.time()
        # Reset GPU memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.batch_start_time = time.time()
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        batch_time = time.time() - self.batch_start_time
        self.batch_times.append(batch_time)
        
        # Update max memory usage
        if torch.cuda.is_available():
            current_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert to GB
            self.max_memory = max(self.max_memory, current_memory)
        
    def on_train_epoch_end(self, trainer, pl_module):
        self.end_time = time.time()
        epoch_time = self.end_time - self.start_time
        self.epoch_times.append(epoch_time)
        
        # Log memory usage
        if torch.cuda.is_available():
            current_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert to GB
            self.max_memory = max(self.max_memory, current_memory)
        
        # Calculate throughput (samples/second)
        avg_batch_time = np.mean(self.batch_times)
        batch_size = trainer.train_dataloader.batch_size
        throughput = batch_size / avg_batch_time
        
        print(f"Epoch {trainer.current_epoch} completed in {epoch_time:.2f}s")
        print(f"Max GPU memory usage: {self.max_memory:.2f} GB")
        print(f"Throughput: {throughput:.2f} samples/second")


def run_benchmark(strategy, num_gpus, batch_size, max_epochs=3):
    """Run a single benchmark configuration"""
    print(f"\n{'='*80}")
    print(f"Running benchmark with strategy={strategy}, num_gpus={num_gpus}, batch_size={batch_size}")
    print(f"{'='*80}\n")
    
    # Data preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download datasets
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Create model
    model = BenchmarkModel()
    
    # Create benchmark callback
    benchmark_callback = BenchmarkCallback()
    
    # Configure trainer
    if num_gpus == 1:
        # For single GPU, don't specify strategy
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            callbacks=[benchmark_callback],
            enable_progress_bar=True,
            logger=False  # Disable logging to keep output clean
        )
    else:
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=min(num_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 1,
            strategy=strategy,
            callbacks=[benchmark_callback],
            enable_progress_bar=True,
            logger=False  # Disable logging to keep output clean
        )
    
    # Create dataloader with specified batch size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Train model
    trainer.fit(model, train_loader)
    
    # Calculate metrics
    avg_epoch_time = np.mean(benchmark_callback.epoch_times)
    avg_batch_time = np.mean(benchmark_callback.batch_times)
    throughput = batch_size / avg_batch_time
    
    # Return benchmark results
    return {
        "strategy": "Single GPU" if num_gpus == 1 else strategy,
        "num_gpus": num_gpus,
        "batch_size": batch_size,
        "avg_epoch_time": avg_epoch_time,
        "max_memory_gb": benchmark_callback.max_memory,
        "throughput": throughput,
        "epoch_times": benchmark_callback.epoch_times,
        "batch_times": benchmark_callback.batch_times
    }


def create_visualizations(results, output_dir="benchmark_results"):
    """Create visualizations from benchmark results"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    
    # Convert results to DataFrame for easier plotting
    df = pd.DataFrame(results)
    
    # Set plot style
    plt.style.use('ggplot')
    sns.set(font_scale=1.2)
    
    # 1. Training Speed Comparison
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x="num_gpus", y="avg_epoch_time", hue="strategy", data=df)
    plt.title("Training Time per Epoch Across GPU Configurations")
    plt.xlabel("Number of GPUs")
    plt.ylabel("Average Epoch Time (seconds)")
    plt.xticks(rotation=0)
    plt.legend(title="Strategy")
    
    # Add batch size annotations
    for i, row in enumerate(df.itertuples()):
        ax.text(i, row.avg_epoch_time + 0.1, f"BS: {row.batch_size}", 
                ha='center', va='bottom', color='black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "images", "training_speed_comparison.png"), dpi=300)
    
    # 2. Memory Usage Comparison
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x="num_gpus", y="max_memory_gb", hue="strategy", data=df)
    plt.title("GPU Memory Usage Across Configurations")
    plt.xlabel("Number of GPUs")
    plt.ylabel("Max GPU Memory Usage (GB)")
    plt.xticks(rotation=0)
    plt.legend(title="Strategy")
    
    # Add batch size annotations
    for i, row in enumerate(df.itertuples()):
        ax.text(i, row.max_memory_gb + 0.1, f"BS: {row.batch_size}", 
                ha='center', va='bottom', color='black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "images", "memory_usage_comparison.png"), dpi=300)
    
    # 3. Throughput Comparison
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x="num_gpus", y="throughput", hue="strategy", data=df)
    plt.title("Training Throughput Across Configurations")
    plt.xlabel("Number of GPUs")
    plt.ylabel("Throughput (samples/second)")
    plt.xticks(rotation=0)
    plt.legend(title="Strategy")
    
    # Add batch size annotations
    for i, row in enumerate(df.itertuples()):
        ax.text(i, row.throughput + 10, f"BS: {row.batch_size}", 
                ha='center', va='bottom', color='black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "images", "throughput_comparison.png"), dpi=300)
    
    # 4. Scaling Efficiency
    # Calculate speedup relative to single GPU
    if 1 in df["num_gpus"].values:
        single_gpu_time = df[df["num_gpus"] == 1]["avg_epoch_time"].values[0]
        df["speedup"] = single_gpu_time / df["avg_epoch_time"]
        df["ideal_speedup"] = df["num_gpus"]
        df["efficiency"] = (df["speedup"] / df["ideal_speedup"]) * 100
        
        plt.figure(figsize=(12, 8))
        ax = sns.lineplot(x="num_gpus", y="speedup", hue="strategy", 
                         style="strategy", markers=True, dashes=False, data=df)
        
        # Add ideal scaling line
        max_gpus = df["num_gpus"].max()
        plt.plot([1, max_gpus], [1, max_gpus], 'k--', label="Ideal Scaling")
        
        plt.title("Scaling Efficiency Across GPU Configurations")
        plt.xlabel("Number of GPUs")
        plt.ylabel("Speedup (relative to single GPU)")
        plt.xticks(df["num_gpus"].unique())
        plt.legend(title="Strategy")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "images", "scaling_efficiency.png"), dpi=300)
    
    # Save results to CSV
    df.to_csv(os.path.join(output_dir, "benchmark_results.csv"), index=False)
    
    # Generate markdown table for README
    markdown_table = df[["strategy", "num_gpus", "batch_size", "avg_epoch_time", 
                        "max_memory_gb", "throughput"]].to_markdown(index=False)
    
    with open(os.path.join(output_dir, "results_table.md"), "w") as f:
        f.write("# Benchmark Results\n\n")
        f.write(markdown_table)
    
    print(f"Visualizations and results saved to {output_dir}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Run multi-GPU training benchmarks")
    parser.add_argument("--strategies", nargs="+", default=["dp", "ddp"], 
                        help="Training strategies to benchmark")
    parser.add_argument("--gpus", nargs="+", type=int, default=[1, 2, 4], 
                        help="Number of GPUs to use")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[64, 128, 256], 
                        help="Batch sizes to benchmark")
    parser.add_argument("--epochs", type=int, default=3, 
                        help="Number of epochs for each benchmark")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", 
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Running benchmarks on CPU.")
        print("For meaningful multi-GPU benchmarks, please run on a system with GPUs.")
    else:
        print(f"Found {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Create timestamp for this benchmark run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    
    # Store all benchmark results
    all_results = []
    
    # Run single GPU benchmark first (as baseline)
    if 1 in args.gpus:
        for batch_size in args.batch_sizes:
            try:
                result = run_benchmark(None, 1, batch_size, args.epochs)
                all_results.append(result)
            except Exception as e:
                print(f"Error running single GPU benchmark with batch_size={batch_size}")
                print(f"Error: {str(e)}")
    
    # Run multi-GPU benchmarks
    for strategy in args.strategies:
        for num_gpus in [g for g in args.gpus if g > 1]:  # Skip single GPU for strategies
            for batch_size in args.batch_sizes:
                try:
                    result = run_benchmark(strategy, num_gpus, batch_size, args.epochs)
                    all_results.append(result)
                except Exception as e:
                    print(f"Error running benchmark with strategy={strategy}, "
                          f"num_gpus={num_gpus}, batch_size={batch_size}")
                    print(f"Error: {str(e)}")
    
    if not all_results:
        print("No benchmark results were collected. Please check for errors above.")
        return
    
    # Create visualizations
    df = create_visualizations(all_results, output_dir)
    
    # Save raw results as JSON
    with open(os.path.join(output_dir, "raw_results.json"), "w") as f:
        # Convert numpy values to Python native types for JSON serialization
        clean_results = []
        for result in all_results:
            clean_result = {}
            for k, v in result.items():
                if isinstance(v, (np.ndarray, list)):
                    clean_result[k] = [float(x) if isinstance(x, np.number) else x for x in v]
                elif isinstance(v, np.number):
                    clean_result[k] = float(v)
                else:
                    clean_result[k] = v
            clean_results.append(clean_result)
        
        json.dump(clean_results, f, indent=2)
    
    print("\nBenchmark Summary:")
    print("-" * 80)
    print(df[["strategy", "num_gpus", "batch_size", "avg_epoch_time", 
              "max_memory_gb", "throughput"]].to_string(index=False))
    print("-" * 80)
    
    # Print recommendations
    if len(df) > 1:  # Only if we have multiple results
        best_throughput = df.loc[df["throughput"].idxmax()]
        best_memory = df.loc[df["max_memory_gb"].idxmin()]
        
        print("\nRecommendations:")
        print(f"- Best throughput: {best_throughput['strategy']} with {best_throughput['num_gpus']} "
              f"GPUs and batch size {best_throughput['batch_size']}")
        print(f"- Most memory efficient: {best_memory['strategy']} with {best_memory['num_gpus']} "
              f"GPUs and batch size {best_memory['batch_size']}")
    
    # Create a README for the benchmark results directory
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write("# Multi-GPU Training Benchmark Results\n\n")
        f.write(f"Benchmark run on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if torch.cuda.is_available():
            f.write("## Hardware Configuration\n\n")
            f.write(f"- Number of GPUs: {torch.cuda.device_count()}\n")
            for i in range(torch.cuda.device_count()):
                f.write(f"- GPU {i}: {torch.cuda.get_device_name(i)}\n")
            f.write("\n")
        
        f.write("## Benchmark Configuration\n\n")
        f.write(f"- Strategies: {', '.join(args.strategies)}\n")
        f.write(f"- GPU Configurations: {', '.join(map(str, args.gpus))}\n")
        f.write(f"- Batch Sizes: {', '.join(map(str, args.batch_sizes))}\n")
        f.write(f"- Epochs per benchmark: {args.epochs}\n\n")
        
        f.write("## Results Summary\n\n")
        f.write(df[["strategy", "num_gpus", "batch_size", "avg_epoch_time", 
                   "max_memory_gb", "throughput"]].to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## Visualizations\n\n")
        f.write("### Training Speed Comparison\n\n")
        f.write("![Training Speed Comparison](./images/training_speed_comparison.png)\n\n")
        
        f.write("### Memory Usage Comparison\n\n")
        f.write("![Memory Usage Comparison](./images/memory_usage_comparison.png)\n\n")
        
        f.write("### Throughput Comparison\n\n")
        f.write("![Throughput Comparison](./images/throughput_comparison.png)\n\n")
        
        if 1 in df["num_gpus"].values:
            f.write("### Scaling Efficiency\n\n")
            f.write("![Scaling Efficiency](./images/scaling_efficiency.png)\n\n")
        
        if len(df) > 1:
            f.write("## Recommendations\n\n")
            f.write(f"- **Best throughput**: {best_throughput['strategy']} with {best_throughput['num_gpus']} "
                  f"GPUs and batch size {best_throughput['batch_size']}\n")
            f.write(f"- **Most memory efficient**: {best_memory['strategy']} with {best_memory['num_gpus']} "
                  f"GPUs and batch size {best_memory['batch_size']}\n\n")
        
        f.write("## How to Run\n\n")
        f.write("To reproduce these benchmarks, run:\n\n")
        f.write("```bash\n")
        f.write(f"python benchmark.py --strategies {' '.join(args.strategies)} --gpus {' '.join(map(str, args.gpus))} --batch-sizes {' '.join(map(str, args.batch_sizes))} --epochs {args.epochs}\n")
        f.write("```\n")
    
    print(f"\nResults saved to {output_dir}")
    print(f"Update your README.md with the table from {output_dir}/results_table.md")
    print(f"See detailed results and visualizations in {output_dir}/README.md")


if __name__ == "__main__":
    main()