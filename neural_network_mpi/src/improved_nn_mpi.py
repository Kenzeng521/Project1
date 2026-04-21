#!/usr/bin/env python3
"""
Improved Distributed Neural Network Training with MPI
- Support for multiple hidden layers with configurable neurons per layer
- Early stopping for convergence
- Both normalized and denormalized RMSE
"""

from mpi4py import MPI
import numpy as np
import pandas as pd
import time
import argparse
import json
import os
from datetime import datetime

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(np.float64)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

class ImprovedDistributedNN:
    def __init__(self, input_size, neurons_list, activation='relu'):
        """
        neurons_list: e.g. [128] for single hidden layer with 128 neurons,
                      or [128,64] for two hidden layers.
        """
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.input_size = input_size
        self.neurons_list = neurons_list[:]  # list of ints
        self.activation = activation

        # Select activation function
        if activation == 'relu':
            self.activation_func = relu
            self.activation_deriv = relu_derivative
        elif activation == 'sigmoid':
            self.activation_func = sigmoid
            self.activation_deriv = sigmoid_derivative
        elif activation == 'tanh':
            self.activation_func = np.tanh
            self.activation_deriv = tanh_derivative
        else:
            raise ValueError("Unsupported activation")

        # construct layer sizes: input -> hidden... -> output(1)
        self.layer_sizes = [self.input_size] + self.neurons_list + [1]
        self.num_layers = len(self.layer_sizes) - 1  # number of parameter layers

        self.history = {
            'loss': [],
            'train_rmse': [],
            'test_rmse': [],
            'train_rmse_denorm': [],
            'test_rmse_denorm': []
        }

        # Early stopping params
        self.best_test_rmse = float('inf')
        self.patience_counter = 0
        self.best_weights = None

        self.init_weights()

    def init_weights(self):
        """Initialize weights for arbitrary number of layers"""
        if self.rank == 0:
            np.random.seed(42)
            W_list = []
            b_list = []
            for i in range(self.num_layers):
                in_dim = self.layer_sizes[i]
                out_dim = self.layer_sizes[i+1]
                # He init for ReLU, Xavier-like for others
                if self.activation == 'relu':
                    W = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / max(1, in_dim))
                else:
                    W = np.random.randn(in_dim, out_dim) * np.sqrt(1.0 / max(1, in_dim))
                b = np.zeros((1, out_dim))
                W_list.append(W)
                b_list.append(b)
        else:
            W_list = None
            b_list = None

        # Broadcast lists (pickle)
        self.weights = self.comm.bcast(W_list, root=0)
        self.biases = self.comm.bcast(b_list, root=0)

    def forward(self, X):
        """Forward propagation for arbitrary depth"""
        a = X
        self.z_list = []
        self.a_list = [a]  # a_list[0] = input
        for i in range(self.num_layers):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            self.z_list.append(z)
            if i < self.num_layers - 1:
                a = self.activation_func(z)
            else:
                # last layer: linear output
                a = z
            self.a_list.append(a)
        return a  # shape (m, 1)

    def compute_local_gradients(self, X, y):
        """Compute gradients on local data.
           NOTE: we return SUM-of-gradients (not averaged) so that after Allreduce
           we can divide by total_samples to get correct mean gradient.
        """
        m = X.shape[0]
        # prepare zero grads
        dw_list = [np.zeros_like(W) for W in self.weights]
        db_list = [np.zeros_like(b) for b in self.biases]
        if m == 0:
            return dw_list, db_list, 0.0, 0

        # forward
        output = self.forward(X)  # (m,1)

        # backprop (without dividing by m yet)
        dz = (output - y.reshape(-1, 1))  # shape (m,1) ; sum over samples will happen later
        for i in range(self.num_layers - 1, -1, -1):
            dw = np.dot(self.a_list[i].T, dz)   # (in_dim, out_dim)
            db = np.sum(dz, axis=0, keepdims=True)  # (1, out_dim)
            dw_list[i] = dw
            db_list[i] = db
            if i > 0:
                da_prev = np.dot(dz, self.weights[i].T)
                dz = da_prev * self.activation_deriv(self.z_list[i-1])

        local_loss = 0.5 * np.sum((output - y.reshape(-1, 1))**2)
        return dw_list, db_list, local_loss, m

    def train_step(self, X_local, y_local, batch_size, lr):
        """Single training step: sample local batch, compute local grads, Allreduce and update"""
        n_local = X_local.shape[0]
        local_batch_size = min(batch_size, n_local)

        # sampling
        if n_local > 0 and local_batch_size > 0:
            indices = np.random.choice(n_local, local_batch_size, replace=(local_batch_size > n_local))
            X_batch = X_local[indices]
            y_batch = y_local[indices]
        else:
            X_batch = X_local
            y_batch = y_local
            local_batch_size = 0

        # compute local grads (SUM-of-gradients over the local batch)
        dw_list_local, db_list_local, local_loss, m = self.compute_local_gradients(X_batch, y_batch)

        # prepare global accumulators and Allreduce per layer
        global_dw_list = []
        global_db_list = []
        for i in range(self.num_layers):
            dw = np.ascontiguousarray(dw_list_local[i], dtype=np.float64)
            db = np.ascontiguousarray(db_list_local[i], dtype=np.float64)
            g_dw = np.zeros_like(dw)
            g_db = np.zeros_like(db)
            self.comm.Allreduce(dw, g_dw, op=MPI.SUM)
            self.comm.Allreduce(db, g_db, op=MPI.SUM)
            global_dw_list.append(g_dw)
            global_db_list.append(g_db)

        # aggregate loss and sample counts
        local_stats = np.array([local_loss, m], dtype=np.float64)
        global_stats = np.zeros(2, dtype=np.float64)
        self.comm.Allreduce(local_stats, global_stats, op=MPI.SUM)

        total_loss = global_stats[0]
        total_samples = int(global_stats[1])

        if total_samples > 0:
            scale = 1.0 / total_samples
            for i in range(self.num_layers):
                # average gradient
                global_dw_list[i] *= scale
                global_db_list[i] *= scale
                # parameter update
                self.weights[i] -= lr * global_dw_list[i]
                self.biases[i] -= lr * global_db_list[i]

            avg_loss = total_loss / total_samples
        else:
            avg_loss = 0.0

        return avg_loss

    def parallel_evaluate(self, X_local, y_local, denorm_stats=None):
        """
        Parallel RMSE computation
        Returns both normalized and denormalized RMSE if stats provided
        """
        if X_local.shape[0] > 0:
            predictions = self.forward(X_local)
            local_se = np.sum((predictions.flatten() - y_local)**2)
            local_count = len(y_local)

            if denorm_stats:
                y_mean, y_std = denorm_stats['target_mean'], denorm_stats['target_std']
                pred_denorm = predictions.flatten() * y_std + y_mean
                y_denorm = y_local * y_std + y_mean
                local_se_denorm = np.sum((pred_denorm - y_denorm)**2)
            else:
                local_se_denorm = 0.0
        else:
            local_se = 0.0
            local_se_denorm = 0.0
            local_count = 0

        # Aggregate
        local_stats = np.array([local_se, local_se_denorm, local_count], dtype=np.float64)
        global_stats = np.zeros(3, dtype=np.float64)
        self.comm.Allreduce(local_stats, global_stats, op=MPI.SUM)

        total_se = global_stats[0]
        total_se_denorm = global_stats[1]
        total_count = int(global_stats[2])

        if total_count > 0:
            rmse = np.sqrt(total_se / total_count)
            rmse_denorm = np.sqrt(total_se_denorm / total_count) if denorm_stats else None
        else:
            rmse = 0.0
            rmse_denorm = None

        return rmse, rmse_denorm

    def save_weights(self):
        """Save current weights as best weights"""
        self.best_weights = {
            'weights': [W.copy() for W in self.weights],
            'biases': [b.copy() for b in self.biases]
        }

    def restore_best_weights(self):
        """Restore best weights"""
        if self.best_weights:
            self.weights = [W.copy() for W in self.best_weights['weights']]
            self.biases = [b.copy() for b in self.best_weights['biases']]

    def train(self, X_train_local, y_train_local, X_test_local, y_test_local,
              max_epochs=500, batch_size=1024, lr=0.001, patience=20,
              min_delta=1e-6, denorm_stats=None, verbose=True):
        """
        Train with early stopping
        """
        if self.rank == 0 and verbose:
            print(f"\nStarting training with early stopping:")
            print(f"  Max epochs: {max_epochs}")
            print(f"  Batch size: {batch_size:,}")
            print(f"  Learning rate: {lr}")
            print(f"  Hidden layers: {len(self.neurons_list)}")
            print(f"  Neurons per layer: {self.neurons_list}")
            print(f"  Activation: {self.activation}")
            print(f"  Processes: {self.size}")
            print(f"  Patience: {patience} epochs")
            print(f"  Min delta: {min_delta}")

            # Data distribution
            total_train = self.comm.reduce(X_train_local.shape[0], op=MPI.SUM, root=0)
            total_test = self.comm.reduce(X_test_local.shape[0], op=MPI.SUM, root=0)
            print(f"  Total training samples: {total_train:,}")
            print(f"  Total test samples: {total_test:,}")
        else:
            self.comm.reduce(X_train_local.shape[0], op=MPI.SUM, root=0)
            self.comm.reduce(X_test_local.shape[0], op=MPI.SUM, root=0)

        start_time = time.time()
        converged = False
        actual_epochs = 0

        for epoch in range(max_epochs):
            loss = self.train_step(X_train_local, y_train_local, batch_size, lr)
            self.history['loss'].append(loss)

            # Evaluate every 5 epochs or at end
            if epoch % 5 == 0 or epoch == max_epochs - 1:
                train_rmse, train_rmse_denorm = self.parallel_evaluate(
                    X_train_local, y_train_local, denorm_stats
                )
                test_rmse, test_rmse_denorm = self.parallel_evaluate(
                    X_test_local, y_test_local, denorm_stats
                )

                self.history['train_rmse'].append(train_rmse)
                self.history['test_rmse'].append(test_rmse)
                if denorm_stats:
                    self.history['train_rmse_denorm'].append(train_rmse_denorm)
                    self.history['test_rmse_denorm'].append(test_rmse_denorm)

                # Check improvement
                if test_rmse < self.best_test_rmse - min_delta:
                    self.best_test_rmse = test_rmse
                    self.patience_counter = 0
                    self.save_weights()
                    improvement_str = " *"
                else:
                    self.patience_counter += 1
                    improvement_str = ""

                if self.rank == 0 and verbose:
                    print(f"Epoch {epoch+1:4d}/{max_epochs}: "
                          f"Loss={loss:.6f}, "
                          f"Train RMSE={train_rmse:.6f}", end="")
                    if denorm_stats:
                        print(f" (${train_rmse_denorm:.2f})", end="")
                    print(f", Test RMSE={test_rmse:.6f}", end="")
                    if denorm_stats:
                        print(f" (${test_rmse_denorm:.2f})", end="")
                    print(f"{improvement_str}")

                # Early stopping
                if self.patience_counter >= patience:
                    if self.rank == 0 and verbose:
                        print(f"\nEarly stopping triggered after {epoch+1} epochs")
                        print(f"Best test RMSE: {self.best_test_rmse:.6f}")
                    converged = True
                    actual_epochs = epoch + 1
                    break

            actual_epochs = epoch + 1

        # restore
        if self.best_weights:
            self.restore_best_weights()
            if self.rank == 0 and verbose:
                print("Restored best weights")

        training_time = time.time() - start_time

        if self.rank == 0 and verbose:
            print(f"\nTraining completed in {training_time:.2f} seconds")
            print(f"Actual epochs run: {actual_epochs}")
            print(f"Converged: {converged}")

        return training_time, actual_epochs, converged

def load_preprocessed_data(file_path, stats_file):
    """Load preprocessed data and statistics"""
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"Loading preprocessed data from {file_path}...")
    df = pd.read_csv(file_path)
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"Loaded {len(df):,} samples")

    with open(stats_file, 'r') as f:
        stats = json.load(f)

    feature_cols = stats['feature_columns']
    target_col = stats['target_column']

    X = df[feature_cols].values.astype(np.float64)
    y = df[target_col].values.astype(np.float64)

    # Normalize with provided stats
    X_mean = np.array([stats['feature_means'][col] for col in feature_cols])
    X_std = np.array([stats['feature_stds'][col] for col in feature_cols])
    X_std[X_std == 0] = 1
    X = (X - X_mean) / X_std

    y_mean = stats['target_mean']
    y_std = stats['target_std']
    y_normalized = (y - y_mean) / y_std

    # Train/test split
    n = len(df)
    n_train = int(0.7 * n)
    np.random.seed(42)
    indices = np.random.permutation(n)
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y_normalized[train_idx]
    y_test = y_normalized[test_idx]

    return X_train, X_test, y_train, y_test, len(feature_cols), stats

def distribute_data(comm, X_train, X_test, y_train, y_test):
    """Distribute data across MPI processes"""
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        n_train = len(X_train)
        n_test = len(X_test)

        # Create chunks for training data
        train_chunks_X = []
        train_chunks_y = []
        chunk_size = n_train // size
        remainder = n_train % size

        start = 0
        for i in range(size):
            end = start + chunk_size + (1 if i < remainder else 0)
            train_chunks_X.append(X_train[start:end].copy())
            train_chunks_y.append(y_train[start:end].copy())
            start = end

        # Create chunks for test data
        test_chunks_X = []
        test_chunks_y = []
        chunk_size = n_test // size
        remainder = n_test % size

        start = 0
        for i in range(size):
            end = start + chunk_size + (1 if i < remainder else 0)
            test_chunks_X.append(X_test[start:end].copy())
            test_chunks_y.append(y_test[start:end].copy())
            start = end
    else:
        train_chunks_X = None
        train_chunks_y = None
        test_chunks_X = None
        test_chunks_y = None

    X_train_local = comm.scatter(train_chunks_X, root=0)
    y_train_local = comm.scatter(train_chunks_y, root=0)
    X_test_local = comm.scatter(test_chunks_X, root=0)
    y_test_local = comm.scatter(test_chunks_y, root=0)

    return X_train_local, y_train_local, X_test_local, y_test_local

def main():
    parser = argparse.ArgumentParser(description='Improved Distributed Neural Network Training')
    parser.add_argument('--data', type=str, default='data/nytaxi2022_preprocessed.csv',
                       help='Path to preprocessed data file')
    parser.add_argument('--stats', type=str, default='data/nytaxi2022_preprocessed_stats.json',
                       help='Path to statistics file')
    parser.add_argument('--max-epochs', type=int, default=500,
                       help='Maximum number of epochs')
    parser.add_argument('--batch-size', type=int, default=4096,
                       help='Batch size (appropriate for large datasets)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    # backwards-compatible (single-layer) hidden-size
    parser.add_argument('--hidden-size', type=int, default=None,
                       help='(legacy) Hidden layer size for single hidden layer')
    # new interface: neurons per layer as comma-separated string
    parser.add_argument('--neurons', type=str, default="128",
                       help='Comma-separated neurons per hidden layer, e.g. "128" or "128,64"')
    parser.add_argument('--hidden-layers', type=int, default=None,
                       help='(optional) number of hidden layers (consistency check)')
    parser.add_argument('--activation', choices=['relu', 'sigmoid', 'tanh'],
                       default='relu', help='Activation function')
    parser.add_argument('--patience', type=int, default=20,
                       help='Patience for early stopping')
    parser.add_argument('--min-delta', type=float, default=1e-6,
                       help='Minimum improvement for early stopping')

    args = parser.parse_args()

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("="*60)
        print("Improved Distributed Neural Network Training")
        print("="*60)
        print(f"MPI Configuration: {size} processes")
        print(f"Data file: {args.data}")

    # load data on rank 0
    if rank == 0:
        X_train, X_test, y_train, y_test, n_features, stats = load_preprocessed_data(
            args.data, args.stats
        )
    else:
        X_train = X_test = y_train = y_test = None
        n_features = None
        stats = None

    # Broadcast feature count and stats
    n_features = comm.bcast(n_features, root=0)
    stats = comm.bcast(stats, root=0)

    # distribute data
    X_train_local, y_train_local, X_test_local, y_test_local = distribute_data(
        comm, X_train, X_test, y_train, y_test
    )

    # parse neurons list (backwards-compatible)
    if args.neurons:
        neurons_list = [int(x) for x in args.neurons.split(',') if x.strip()]
    elif args.hidden_size is not None:
        neurons_list = [args.hidden_size]
    else:
        neurons_list = [128]

    # consistency check with hidden-layers if provided
    if args.hidden_layers is not None and args.hidden_layers != len(neurons_list):
        if rank == 0:
            print(f"Warning: --hidden-layers ({args.hidden_layers}) != number of items in --neurons ({len(neurons_list)}). Ignoring --hidden-layers and using --neurons.")
        # we'll just use neurons_list

    # create and train model
    model = ImprovedDistributedNN(
        input_size=n_features,
        neurons_list=neurons_list,
        activation=args.activation
    )

    training_time, actual_epochs, converged = model.train(
        X_train_local, y_train_local,
        X_test_local, y_test_local,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        patience=args.patience,
        min_delta=args.min_delta,
        denorm_stats=stats,
        verbose=(rank == 0)
    )

    if rank == 0:
        final_train_rmse = model.history['train_rmse'][-1] if model.history['train_rmse'] else 0
        final_test_rmse = model.history['test_rmse'][-1] if model.history['test_rmse'] else 0

        if stats:
            final_train_rmse_denorm = model.history['train_rmse_denorm'][-1] if model.history['train_rmse_denorm'] else 0
            final_test_rmse_denorm = model.history['test_rmse_denorm'][-1] if model.history['test_rmse_denorm'] else 0
        else:
            final_train_rmse_denorm = 0
            final_test_rmse_denorm = 0

        print("\n" + "="*60)
        print("TRAINING RESULTS")
        print("="*60)
        print(f"Configuration:")
        print(f"  Activation: {args.activation}")
        print(f"  Hidden layers: {len(neurons_list)}")
        print(f"  Neurons per layer: {neurons_list}")
        print(f"  Batch size: {args.batch_size:,}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Processes: {size}")
        print(f"\nTraining Summary:")
        print(f"  Epochs run: {actual_epochs}/{args.max_epochs}")
        print(f"  Converged: {converged}")
        print(f"  Training time: {training_time:.2f} seconds")
        print(f"\nPerformance Metrics:")
        print(f"  Normalized RMSE:")
        print(f"    Train: {final_train_rmse:.6f}")
        print(f"    Test: {final_test_rmse:.6f}")
        if stats:
            print(f"  Denormalized RMSE (actual dollars):")
            print(f"    Train: ${final_train_rmse_denorm:.2f}")
            print(f"    Test: ${final_test_rmse_denorm:.2f}")

        os.makedirs('results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        neurons_str = "-".join(map(str, neurons_list))
        result_file = f'results/result_{args.activation}_layers{len(neurons_list)}_neurons{neurons_str}_b{args.batch_size}_p{size}_{timestamp}.json'

        results = {
            'activation': args.activation,
            'neurons_list': neurons_list,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'max_epochs': args.max_epochs,
            'actual_epochs': actual_epochs,
            'converged': converged,
            'num_processes': size,
            'train_rmse': final_train_rmse,
            'test_rmse': final_test_rmse,
            'train_rmse_denorm': final_train_rmse_denorm,
            'test_rmse_denorm': final_test_rmse_denorm,
            'training_time': training_time,
            'patience': args.patience,
            'min_delta': args.min_delta,
            'loss_history': model.history['loss'],
            'train_rmse_history': model.history['train_rmse'],
            'test_rmse_history': model.history['test_rmse'],
            'train_rmse_denorm_history': model.history['train_rmse_denorm'],
            'test_rmse_denorm_history': model.history['test_rmse_denorm']
        }

        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {result_file}")

if __name__ == "__main__":
    main()
