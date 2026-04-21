Distributed Neural Network Training with MPI

Overview
This project implements a distributed neural network training system using MPI (Message Passing Interface) for parallel computation. It demonstrates data-parallel stochastic gradient descent (SGD) for training a single-hidden-layer neural network on the NYC Taxi 2022 dataset, achieving significant speedup through parallel processing.

Prerequisites
System Requirements

Python 3.7 or higher
MPI implementation (MPICH or OpenMPI)
Multi-core processor (4+ cores recommended)

Python Dependencies
bashpip install mpi4py numpy pandas matplotlib
Installing MPI

Ubuntu/Debian:
bashsudo apt-get update
sudo apt-get install mpich

macOS:
bashbrew install mpich

Verify Installation:
bashmpirun --version

Project Structure
NEURAL_NETWORK_MPI/
├── src/
│   ├── neural_network_mpi.py      # Main distributed training implementation
│   ├── analyze_improved_results.py # Results analysis and visualization
│   ├── mpi_test.py                # MPI functionality test
│   └── mpi_data_distribution.py   # Data distribution test
├── data/
│   └── nytaxi2022.csv            # NYC Taxi dataset (or mock data)
├── results/                       # Experiment results (JSON files)
├── logs/                          # Training logs
└── run_full_experiments.sh       # Automated experiment script

Quick Start

1. Basic Test Run
bash# Test with 4 processes, small dataset
mpirun -np 4 python src/neural_network_mpi.py \
    --epochs 10 \
    --batch-size 64 \
    --activation relu \
    --hidden-size 50 \
    --sample-size 10000

2. Run Full Experiments
bash# Make script executable
chmod +x run_full_experiments.sh

# Run all experiments (3 activations × 5 batch sizes × multiple processes)
./run_full_experiments.sh

3. Analyze Results
bash# Generate comprehensive analysis report and visualizations
python src/analyze_improved_results.py
Algorithm Overview
Neural Network Architecture
The system implements a feedforward neural network with:

Input Layer: 9 features from NYC taxi data
Hidden Layer: Variable size (30-100 neurons) with configurable activation
Output Layer: Single neuron for regression (fare prediction)

Stochastic Gradient Descent (SGD)
The training process follows these steps:

Initialize: Model parameters θ = (W₁, b₁, W₂, b₂)
Iterate: For each epoch:

Sample random mini-batch from local data
Compute forward pass: f(x; θ)
Calculate loss: R(θ) = ½ Σ(f(xᵢ; θ) - yᵢ)²
Compute gradients via backpropagation
Update parameters: θₖ₊₁ = θₖ - η∇R(θₖ)



Distributed Computing Strategy
Data Parallelism
The system employs data-parallel training where:

Data Distribution: Training data is split equally among P processes
Model Synchronization: All processes maintain identical model parameters
Gradient Aggregation: Local gradients are averaged across all processes

MPI Communication Pattern
Process 0        Process 1        Process 2        Process 3
    |                |                |                |
    |-- Broadcast initial weights -->|                |
    |                |                |                |
[Local Data]    [Local Data]    [Local Data]    [Local Data]
    ↓                ↓                ↓                ↓
[Compute]       [Compute]       [Compute]       [Compute]
[Gradient]      [Gradient]      [Gradient]      [Gradient]
    |                |                |                |
    |<------------ Allreduce (Sum) --------------->|
    |                |                |                |
[Update θ]      [Update θ]      [Update θ]      [Update θ]
Scalability Features

Linear Data Scaling: Each process handles N/P samples
Constant Memory: Memory per process remains constant as P increases
Communication Efficiency: Only gradients are communicated (not data)
Load Balancing: Work is evenly distributed with remainder handling

Detailed Usage
Command Line Arguments
bashpython src/neural_network_mpi.py [options]

Options:
  --data PATH           Path to CSV data file (default: data/nytaxi2022.csv)
  --epochs N            Number of training epochs (default: 50)
  --batch-size N        Mini-batch size (default: 64)
  --learning-rate LR    Learning rate (default: 0.001)
  --hidden-size N       Hidden layer size (default: 50)
  --activation ACT      Activation function: relu|sigmoid|tanh (default: relu)
  --sample-size N       Number of samples to use (default: 100000)
Running with Different Process Counts
bash# Single process (baseline)
mpirun -np 1 python src/neural_network_mpi.py

# 2 processes
mpirun -np 2 python src/neural_network_mpi.py

# 4 processes (recommended)
mpirun -np 4 python src/neural_network_mpi.py

# 8 processes (for larger datasets)
mpirun -np 8 python src/neural_network_mpi.py
Performance Optimization
Computational Optimizations

He Initialization: Improved convergence for ReLU networks
Batch Processing: Reduces communication overhead
Vectorized Operations: NumPy array operations for efficiency

Communication Optimizations

Collective Operations: MPI Allreduce for efficient gradient aggregation
Minimal Data Transfer: Only gradients are communicated between iterations
Contiguous Arrays: Memory layout optimization for MPI operations

Scaling Results
Expected speedup with P processes:

2 processes: ~1.8x speedup
4 processes: ~3.2x speedup
8 processes: ~5.5x speedup

Efficiency decreases with more processes due to communication overhead.
Experiment Configuration
The automated experiments test:

Activation Functions: ReLU, Sigmoid, Tanh
Batch Sizes: 32, 64, 128, 256, 512
Hidden Layer Sizes: 30-100 neurons (optimized per activation)
Process Counts: 1, 2, 4, 8 processes
Learning Rates: Tuned for each activation function

Output Files
Results Directory

result_*.json: Individual experiment results with metrics
comprehensive_analysis.png: Performance visualization plots
experiment_report.md: Detailed markdown report

Log Files

logs/exp_*.log: Training output for each experiment

Troubleshooting
Common Issues

MPI Not Found

bash   # Verify MPI installation
   which mpirun
   mpirun --version

Data File Missing

The system automatically generates mock data if the CSV file is not found
To create test data: python create_test_data.py


Memory Issues

Reduce --sample-size parameter
Use fewer processes
Reduce --hidden-size


Slow Performance

Ensure MPI is using all available cores
Check network latency if using multiple machines
Verify NumPy is using optimized BLAS libraries



Key Implementation Features
Fault Tolerance

Automatic fallback to mock data if real data unavailable
Graceful handling of edge cases (empty partitions, division by zero)
Numerical stability (gradient clipping, overflow prevention)

Monitoring and Analysis

Real-time training progress display
Comprehensive metrics tracking (loss, RMSE, time)
Automated result analysis and visualization
Detailed experiment reports

References

MPI4PY Documentation: https://mpi4py.readthedocs.io/
NYC Taxi Dataset: https://www.kaggle.com/datasets/diishasiing/revenue-for-cab-drivers
Distributed SGD Theory: "Large Scale Distributed Deep Networks" (Dean et al., 2012)

License
This project is developed for educational purposes as part of DSA5208 coursework.