import numpy as np
import pandas as pd
from mpi4py import MPI
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
import os
import argparse
import sys
import gc

class MPINeuralNetwork:
    def __init__(self, input_size, hidden_size, learning_rate=0.001, activation='relu'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.activation = activation
        
        self.W1 = np.random.randn(input_size, hidden_size).astype(np.float32) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.W2 = np.random.randn(hidden_size, 1).astype(np.float32) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(1, dtype=np.float32)
        
        self.train_losses = []
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(np.float32)
    
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def leaky_relu_derivative(self, x, alpha=0.01):
        dx = np.ones_like(x, dtype=np.float32)
        dx[x < 0] = alpha
        return dx
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - np.tanh(x)**2
    
    def activate(self, x):
        if self.activation == 'relu':
            return self.relu(x)
        elif self.activation == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation == 'leaky_relu':
            return self.leaky_relu(x)
        elif self.activation == 'tanh':
            return self.tanh(x)
    
    def activate_derivative(self, x):
        if self.activation == 'relu':
            return self.relu_derivative(x)
        elif self.activation == 'sigmoid':
            return self.sigmoid_derivative(x)
        elif self.activation == 'leaky_relu':
            return self.leaky_relu_derivative(x)
        elif self.activation == 'tanh':
            return self.tanh_derivative(x)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activate(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.z2 
        return self.a2
    
    def compute_loss(self, y_true, y_pred):
        return 0.5 * np.mean((y_pred - y_true) ** 2)
    
    def backward(self, X, y_true, y_pred):
        m = X.shape[0]

        dz2 = y_pred - y_true
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.mean(dz2, axis=0)
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.activate_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.mean(dz1, axis=0)
        
        return dW1, db1, dW2, db2
    
    def get_parameters_flat(self):
        return np.concatenate([
            self.W1.flatten(),
            self.b1.flatten(),
            self.W2.flatten(),
            self.b2.flatten()
        ])
    
    def set_parameters_flat(self, params):
        idx = 0
        
        # W1
        w1_size = self.input_size * self.hidden_size
        self.W1 = params[idx:idx+w1_size].reshape(self.input_size, self.hidden_size)
        idx += w1_size
        
        # b1
        self.b1 = params[idx:idx+self.hidden_size]
        idx += self.hidden_size
        
        # W2
        w2_size = self.hidden_size
        self.W2 = params[idx:idx+w2_size].reshape(self.hidden_size, 1)
        idx += w2_size
        
        # b2
        self.b2 = params[idx:idx+1]

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Memory-Optimized MPI Neural Network')
    
    parser.add_argument('--hidden_size', type=int, default=128,  
                       help='Number of neurons in hidden layer (default: 128)')
    parser.add_argument('--activation', type=str, default='relu', 
                       choices=['relu', 'sigmoid', 'tanh', 'leaky_relu'],
                       help='Activation function (default: relu)')
    
    parser.add_argument('--batch_size', type=int, default=256, 
                       help='Batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps (default: 4)')
    
    parser.add_argument('--data_file', type=str, default='process3_data.csv',
                       help='Path to processed data file (default: process3_data.csv)')
    parser.add_argument('--test_size', type=float, default=0.3, 
                       help='Test set proportion (default: 0.2)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--max_samples', type=int, default=None,  
                       help='Maximum number of samples to use (for memory limitation)')
    
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results (default: results)')
    parser.add_argument('--save_model', action='store_true',
                       help='Save trained model parameters')
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level (0: quiet, 1: normal, 2: detailed)')
    
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Custom experiment name for output files')
    
    return parser.parse_args()

def load_and_split_data(rank, size, args):
    """内存优化的数据加载和分割"""
    if rank == 0:
        if args.verbose >= 1:
            print(f"Loading data from {args.data_file}...")
        
        try:
            chunk_size = 10000
            chunks = []
            for chunk in pd.read_csv(args.data_file, chunksize=chunk_size):
                chunks.append(chunk)
                if args.max_samples and sum(len(c) for c in chunks) >= args.max_samples:
                    break
            
            df = pd.concat(chunks, ignore_index=True)
            if args.max_samples:
                df = df.head(args.max_samples)
            
            del chunks
            gc.collect()
            
        except FileNotFoundError:
            print(f"Error: Data file '{args.data_file}' not found!")
            print("Please run preprocess.py first or specify correct data file path.")
            sys.exit(1)
        
        feature_cols = ["passenger_count", "trip_distance", "RatecodeID", 
                       "PULocationID", "DOLocationID", "payment_type", "extra",
                       "trip_duration"]
        
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing columns in data file: {missing_cols}")
            sys.exit(1)
        
        if "total_amount" not in df.columns:
            print("Error: Target column 'total_amount' not found in data file!")
            sys.exit(1)
        
        X = df[feature_cols].values.astype(np.float32)
        y = df["total_amount"].values.astype(np.float32).reshape(-1, 1)
        
        del df
        gc.collect()
        
        if args.verbose >= 1:
            print(f"Dataset shape: X={X.shape}, y={y.shape}")
            print(f"Memory usage: X={X.nbytes/1024/1024:.1f}MB, y={y.nbytes/1024/1024:.1f}MB")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_seed
        )
        
        del X, y
        gc.collect()
        
        if args.verbose >= 1:
            print(f"Training set: X={X_train.shape}, y={y_train.shape}")
            print(f"Test set: X={X_test.shape}, y={y_test.shape}")
        
        n_train = X_train.shape[0]
        indices = np.array_split(np.arange(n_train), size)
        
        local_data = []
        test_data = []
        for i in range(size):

            train_idx = indices[i]
            local_data.append((X_train[train_idx], y_train[train_idx]))
            
            test_indices = np.array_split(np.arange(X_test.shape[0]), size)
            test_idx = test_indices[i]
            test_data.append((X_test[test_idx], y_test[test_idx]))
            
            if args.verbose >= 2:
                print(f"Process {i}: train={len(train_idx)}, test={len(test_idx)} samples")
        
        del X_train, y_train, X_test, y_test
        gc.collect()
        
    else:
        local_data = None
        test_data = None
    
    local_X_train, local_y_train = comm.scatter(local_data, root=0)
    local_X_test, local_y_test = comm.scatter(test_data, root=0)
    
    return local_X_train, local_y_train, local_X_test, local_y_test

def train_mpi_neural_network(args):
    
    local_X_train, local_y_train, local_X_test, local_y_test = load_and_split_data(rank, size, args)
    
    if rank == 0 and args.verbose >= 1:
        print(f"\n{'='*50}")
        print(f"Training Configuration:")
        print(f"  Hidden Size: {args.hidden_size}")
        print(f"  Activation: {args.activation}")
        print(f"  Batch Size: {args.batch_size}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Learning Rate: {args.learning_rate}")
        print(f"  Gradient Accumulation: {args.gradient_accumulation_steps}")
        print(f"  MPI Processes: {size}")
        print(f"{'='*50}")
    input_size = local_X_train.shape[1]
    nn = MPINeuralNetwork(input_size, args.hidden_size, args.learning_rate, args.activation)
    
    if rank == 0:
        initial_params = nn.get_parameters_flat()
    else:
        initial_params = None
    
    initial_params = comm.bcast(initial_params, root=0)
    nn.set_parameters_flat(initial_params)
    
    train_losses = []
    n_local_samples = local_X_train.shape[0]
    
    accumulated_dW1 = np.zeros_like(nn.W1, dtype=np.float32)
    accumulated_db1 = np.zeros_like(nn.b1, dtype=np.float32)
    accumulated_dW2 = np.zeros_like(nn.W2, dtype=np.float32)
    accumulated_db2 = np.zeros_like(nn.b2, dtype=np.float32)
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        indices = np.random.permutation(n_local_samples)
        
        epoch_losses = []
        n_batches = 0
        accumulation_count = 0
        
        for i in range(0, n_local_samples, args.batch_size):
            batch_indices = indices[i:i+args.batch_size]
            batch_X = local_X_train[batch_indices]
            batch_y = local_y_train[batch_indices]
            
            y_pred = nn.forward(batch_X)
            
            loss = nn.compute_loss(batch_y, y_pred)
            epoch_losses.append(loss)
            n_batches += 1
            
            dW1, db1, dW2, db2 = nn.backward(batch_X, batch_y, y_pred)
            
            accumulated_dW1 += dW1
            accumulated_db1 += db1
            accumulated_dW2 += dW2
            accumulated_db2 += db2
            accumulation_count += 1
            
            if (accumulation_count >= args.gradient_accumulation_steps or 
                i + args.batch_size >= n_local_samples):
                
                accumulated_dW1 /= accumulation_count
                accumulated_db1 /= accumulation_count
                accumulated_dW2 /= accumulation_count
                accumulated_db2 /= accumulation_count
                
                dW1_global = np.zeros_like(accumulated_dW1, dtype=np.float32)
                db1_global = np.zeros_like(accumulated_db1, dtype=np.float32)
                dW2_global = np.zeros_like(accumulated_dW2, dtype=np.float32)
                db2_global = np.zeros_like(accumulated_db2, dtype=np.float32)
                
                comm.Allreduce(accumulated_dW1, dW1_global, op=MPI.SUM)
                comm.Allreduce(accumulated_db1, db1_global, op=MPI.SUM)
                comm.Allreduce(accumulated_dW2, dW2_global, op=MPI.SUM)
                comm.Allreduce(accumulated_db2, db2_global, op=MPI.SUM)
                
                dW1_global /= size
                db1_global /= size
                dW2_global /= size
                db2_global /= size
                
                nn.W1 -= args.learning_rate * dW1_global
                nn.b1 -= args.learning_rate * db1_global
                nn.W2 -= args.learning_rate * dW2_global
                nn.b2 -= args.learning_rate * db2_global
                
                accumulated_dW1.fill(0)
                accumulated_db1.fill(0)
                accumulated_dW2.fill(0)
                accumulated_db2.fill(0)
                accumulation_count = 0
                
                gc.collect()
        
        if len(epoch_losses) > 0:
            avg_local_loss = np.mean(epoch_losses)
            
            global_loss = comm.reduce(avg_local_loss, op=MPI.SUM, root=0)
            
            if rank == 0:
                global_loss /= size
                train_losses.append(global_loss)
                
                epoch_time = time.time() - epoch_start
                if args.verbose >= 1:
                    print(f"Epoch {epoch+1:3d}/{args.epochs}, "
                          f"Loss: {global_loss:.6f}, "
                          f"Batches: {n_batches:3d}, "
                          f"Time: {epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    
    if rank == 0 and args.verbose >= 1:
        print(f"\nTraining completed in {total_time:.2f} seconds")
        print(f"Average time per epoch: {total_time/args.epochs:.2f} seconds")
    
    train_rmse = compute_rmse_parallel(nn, local_X_train, local_y_train, "Training", args.verbose)
    test_rmse = compute_rmse_parallel(nn, local_X_test, local_y_test, "Test", args.verbose)
    
    return nn, train_losses, train_rmse, test_rmse, total_time

def compute_rmse_parallel(nn, X, y, dataset_name, verbose=1):
    """内存高效的并行RMSE计算"""
    if X is None or len(X) == 0:
        local_mse = 0.0
        local_samples = 0
    else:
        batch_size = 1000
        total_mse = 0.0
        total_samples = 0
        
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            y_pred = nn.forward(batch_X)
            batch_mse = np.sum((y_pred - batch_y) ** 2)
            
            total_mse += batch_mse
            total_samples += len(batch_X)
        
        local_mse = total_mse
        local_samples = total_samples

    global_mse = comm.reduce(local_mse, op=MPI.SUM, root=0)
    global_samples = comm.reduce(local_samples, op=MPI.SUM, root=0)
    
    if rank == 0 and global_samples > 0:
        rmse = np.sqrt(global_mse / global_samples)
        if verbose >= 1:
            print(f"{dataset_name} RMSE: {rmse:.6f}")
        return rmse
    
    return None

def save_results(nn, train_losses, train_rmse, test_rmse, total_time, args):
    """保存结果"""
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        
        if args.experiment_name:
            exp_name = args.experiment_name
        else:
            exp_name = (f"h{args.hidden_size}_{args.activation}_"
                       f"b{args.batch_size}_e{args.epochs}_"
                       f"lr{args.learning_rate:.0e}_opt")
        
        results = {
            'experiment_name': exp_name,
            'hidden_size': args.hidden_size,
            'activation': args.activation,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'gradient_accumulation_steps': args.gradient_accumulation_steps,
            'train_losses': train_losses,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'training_time': total_time,
            'mpi_processes': size,
            'final_loss': train_losses[-1] if train_losses else None
        }
        
        results_file = os.path.join(args.output_dir, f"results_{exp_name}.pkl")
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        if args.save_model:
            model_file = os.path.join(args.output_dir, f"model_{exp_name}.pkl")
            model_data = {
                'W1': nn.W1,
                'b1': nn.b1,
                'W2': nn.W2,
                'b2': nn.b2,
                'input_size': nn.input_size,
                'hidden_size': nn.hidden_size,
                'activation': nn.activation
            }
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)
            if args.verbose >= 1:
                print(f"Model saved to {model_file}")
        
        if train_losses:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(train_losses)+1), train_losses, 'b-', linewidth=2, marker='o', markersize=4)
            plt.title(f'Training Loss History (Memory Optimized)\n{exp_name}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            
            loss_plot_file = os.path.join(args.output_dir, f'training_loss_{exp_name}.png')
            plt.savefig(loss_plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            if args.verbose >= 1:
                print(f"Training loss plot saved to {loss_plot_file}")
        
        if args.verbose >= 1:
            print(f"\n{'='*50}")
            print(f"MEMORY-OPTIMIZED EXPERIMENT SUMMARY")
            print(f"{'='*50}")
            print(f"Experiment Name: {exp_name}")
            print(f"Configuration:")
            print(f"  - Hidden Size: {args.hidden_size}")
            print(f"  - Activation: {args.activation}")
            print(f"  - Batch Size: {args.batch_size}")
            print(f"  - Epochs: {args.epochs}")
            print(f"  - Learning Rate: {args.learning_rate}")
            print(f"  - Gradient Accumulation: {args.gradient_accumulation_steps}")
            print(f"  - MPI Processes: {size}")
            print(f"Results:")
            print(f"  - Training RMSE: {train_rmse:.6f}")
            print(f"  - Test RMSE: {test_rmse:.6f}")
            print(f"  - Final Loss: {train_losses[-1]:.6f}" if train_losses else "  - Final Loss: N/A")
            print(f"  - Training Time: {total_time:.2f}s")
            print(f"  - Time per Epoch: {total_time/args.epochs:.2f}s")
            print(f"Results saved to {results_file}")
        
        return results_file

if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        args = parse_arguments()
        if args.verbose >= 1:
            print(f"Memory-Optimized MPI Neural Network Training")
            print(f"Running with {size} MPI processes")
    else:
        args = None
    
    args = comm.bcast(args, root=0)
    
    try:
        nn, train_losses, train_rmse, test_rmse, total_time = train_mpi_neural_network(args)
        
        results_file = save_results(nn, train_losses, train_rmse, test_rmse, total_time, args)
        
        if rank == 0 and args.verbose >= 1:
            print(f"\nMemory-optimized experiment completed successfully!")
            
    except Exception as e:
        if rank == 0:
            print(f"Error during training: {str(e)}")
            import traceback
            traceback.print_exc()
        sys.exit(1)