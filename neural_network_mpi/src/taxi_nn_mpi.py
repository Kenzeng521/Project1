#!/usr/bin/env python3
"""
Distributed Neural Network for NYC Taxi Fare Prediction with MPI
- Automatically compares two architectures: 512,256 vs 256,128
- Multi-layer neural network with embedding layers for categorical features
- SGD implementation with embeddings for categorical variables
- RMSE tracking for training history
"""

from mpi4py import MPI
import numpy as np
import pandas as pd
import time
import argparse
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt

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

class EmbeddingLayer:
    """Simple embedding layer for categorical features"""
    def __init__(self, num_embeddings, embedding_dim, feature_name):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.feature_name = feature_name
        
        # Initialize embedding weights with small random values
        self.weights = np.random.randn(num_embeddings, embedding_dim) * 0.1
        
    def forward(self, indices):
        """Forward pass: lookup embeddings for given indices"""
        # Ensure indices are within bounds
        indices = np.clip(indices.astype(int), 0, self.num_embeddings - 1)
        return self.weights[indices]
    
    def backward(self, indices, grad_output):
        """Backward pass: compute gradients for embedding weights"""
        indices = np.clip(indices.astype(int), 0, self.num_embeddings - 1)
        grad_weights = np.zeros_like(self.weights)
        
        # Accumulate gradients for each embedding
        for i, idx in enumerate(indices):
            grad_weights[idx] += grad_output[i]
        
        return grad_weights

class MultiLayerTaxiNN:
    def __init__(self, numerical_features_size, categorical_config, hidden_layers=[256, 128], activation='relu'):
        """
        Multi-layer Neural Network for Taxi Fare Prediction with Embeddings
        
        Args:
            numerical_features_size: Number of numerical features
            categorical_config: Dict with categorical feature info
            hidden_layers: List of neurons in each hidden layer
            activation: Activation function ('relu', 'sigmoid', 'tanh')
        """
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.numerical_features_size = numerical_features_size
        self.categorical_config = categorical_config
        self.hidden_layers = hidden_layers
        self.activation = activation
        
        # Create embedding layers for categorical features
        self.embedding_layers = {}
        self.total_embedding_size = 0
        
        for feature_name, config in categorical_config.items():
            num_categories = config['num_categories']
            embedding_dim = config['embedding_dim']
            
            self.embedding_layers[feature_name] = EmbeddingLayer(
                num_categories, embedding_dim, feature_name
            )
            self.total_embedding_size += embedding_dim
        
        # Calculate total input size after embeddings
        self.final_input_size = numerical_features_size + self.total_embedding_size
        
        # Build layer sizes: input -> hidden_layers -> output(1)
        self.layer_sizes = [self.final_input_size] + hidden_layers + [1]
        self.num_layers = len(self.layer_sizes) - 1

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
            raise ValueError(f"Unsupported activation: {activation}")

        # Training history
        self.history = {
            'loss': [],
            'train_rmse': [],
            'test_rmse': [],
            'epochs': []
        }

        # Early stopping
        self.best_test_rmse = float('inf')
        self.patience_counter = 0
        self.best_weights = None
        self.best_embeddings = None

        self.init_weights()

    def init_weights(self):
        """Initialize weights for multi-layer network"""
        if self.rank == 0:
            np.random.seed(42)
            
            self.weights = []
            self.biases = []
            
            for i in range(self.num_layers):
                in_size = self.layer_sizes[i]
                out_size = self.layer_sizes[i + 1]
                
                # Weight initialization
                if self.activation == 'relu':
                    # He initialization for ReLU
                    W = np.random.randn(in_size, out_size) * np.sqrt(2.0 / in_size)
                else:
                    # Xavier initialization for sigmoid/tanh
                    W = np.random.randn(in_size, out_size) * np.sqrt(1.0 / in_size)
                
                b = np.zeros((1, out_size))
                
                self.weights.append(W)
                self.biases.append(b)
        else:
            self.weights = None
            self.biases = None

        # Broadcast weights to all processes
        self.weights = self.comm.bcast(self.weights, root=0)
        self.biases = self.comm.bcast(self.biases, root=0)
        
        # Broadcast embedding weights
        for feature_name, embedding_layer in self.embedding_layers.items():
            embedding_layer.weights = self.comm.bcast(embedding_layer.weights, root=0)

    def prepare_input(self, X_numerical, X_categorical):
        """Prepare input by applying embeddings to categorical features"""
        batch_size = X_numerical.shape[0]
        
        # Start with numerical features
        embedded_features = [X_numerical]
        
        # Apply embeddings to categorical features
        for feature_name, feature_indices in X_categorical.items():
            embedding_layer = self.embedding_layers[feature_name]
            embedded = embedding_layer.forward(feature_indices)
            embedded_features.append(embedded)
        
        # Concatenate all features
        return np.concatenate(embedded_features, axis=1)

    def forward(self, X_numerical, X_categorical):
        """Forward propagation through embeddings and neural network"""
        # Apply embeddings and concatenate features
        X_embedded = self.prepare_input(X_numerical, X_categorical)
        
        self.activations = [X_embedded]  # Store activations for backprop
        self.z_values = []               # Store z values for backprop
        
        a = X_embedded
        for i in range(self.num_layers):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            if i < self.num_layers - 1:  # Hidden layers
                a = self.activation_func(z)
            else:  # Output layer (linear for regression)
                a = z
                
            self.activations.append(a)
        
        return a

    def compute_local_gradients(self, X_numerical, X_categorical, y):
        """Compute gradients for multi-layer network with embeddings"""
        m = X_numerical.shape[0]
        if m == 0:
            zero_weights = [np.zeros_like(w) for w in self.weights]
            zero_biases = [np.zeros_like(b) for b in self.biases]
            zero_embeddings = {name: np.zeros_like(emb.weights) 
                             for name, emb in self.embedding_layers.items()}
            return zero_weights, zero_biases, zero_embeddings, 0.0, 0

        # Forward pass
        output = self.forward(X_numerical, X_categorical)
        
        # Compute loss (MSE)
        loss = 0.5 * np.mean((output - y.reshape(-1, 1))**2)
        
        # Backward pass for neural network layers
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Start from output layer
        delta = (output - y.reshape(-1, 1)) / m
        
        # Backpropagate through neural network layers
        for i in range(self.num_layers - 1, -1, -1):
            # Compute gradients for current layer
            dW[i] = np.dot(self.activations[i].T, delta)
            db[i] = np.sum(delta, axis=0, keepdims=True)
            
            # Compute delta for previous layer (if not input layer)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_deriv(self.z_values[i-1])
        
        # Compute gradients for embeddings
        embedding_gradients = {}
        input_grad = delta  # Gradient w.r.t input
        
        # Split gradient back to numerical and categorical parts
        start_idx = self.numerical_features_size
        
        for feature_name, feature_indices in X_categorical.items():
            embedding_layer = self.embedding_layers[feature_name]
            end_idx = start_idx + embedding_layer.embedding_dim
            
            # Extract gradient for this embedding
            grad_embedding_output = input_grad[:, start_idx:end_idx]
            
            # Compute embedding weight gradients
            embedding_gradients[feature_name] = embedding_layer.backward(
                feature_indices, grad_embedding_output
            )
            
            start_idx = end_idx
        
        return dW, db, embedding_gradients, loss, m

    def train_step(self, X_numerical_local, X_categorical_local, y_local, batch_size, lr):
        """Single SGD training step with embeddings"""
        n_local = X_numerical_local.shape[0]
        
        if n_local > 0:
            # Sample local batch
            if batch_size < n_local:
                indices = np.random.choice(n_local, batch_size, replace=False)
                X_num_batch = X_numerical_local[indices]
                X_cat_batch = {name: X_categorical_local[name][indices] 
                              for name in X_categorical_local}
                y_batch = y_local[indices]
            else:
                X_num_batch = X_numerical_local
                X_cat_batch = X_categorical_local
                y_batch = y_local
        else:
            X_num_batch = X_numerical_local
            X_cat_batch = X_categorical_local
            y_batch = y_local

        # Compute local gradients
        dW_local, db_local, emb_grad_local, local_loss, local_m = self.compute_local_gradients(
            X_num_batch, X_cat_batch, y_batch
        )

        # Allreduce neural network gradients
        dW_global = []
        db_global = []
        
        for i in range(self.num_layers):
            dW_local_cont = np.ascontiguousarray(dW_local[i], dtype=np.float64)
            db_local_cont = np.ascontiguousarray(db_local[i], dtype=np.float64)
            
            dW_glob = np.zeros_like(dW_local_cont)
            db_glob = np.zeros_like(db_local_cont)
            
            self.comm.Allreduce(dW_local_cont, dW_glob, op=MPI.SUM)
            self.comm.Allreduce(db_local_cont, db_glob, op=MPI.SUM)
            
            dW_global.append(dW_glob)
            db_global.append(db_glob)

        # Allreduce embedding gradients
        emb_grad_global = {}
        for feature_name, grad in emb_grad_local.items():
            grad_cont = np.ascontiguousarray(grad, dtype=np.float64)
            grad_glob = np.zeros_like(grad_cont)
            self.comm.Allreduce(grad_cont, grad_glob, op=MPI.SUM)
            emb_grad_global[feature_name] = grad_glob

        # Allreduce loss and sample count
        local_stats = np.array([local_loss * local_m, local_m], dtype=np.float64)
        global_stats = np.zeros(2, dtype=np.float64)
        self.comm.Allreduce(local_stats, global_stats, op=MPI.SUM)

        total_loss_sum = global_stats[0]
        total_samples = int(global_stats[1])

        if total_samples > 0:
            # Average gradients across processes
            for i in range(self.num_layers):
                dW_global[i] /= self.size
                db_global[i] /= self.size

            for feature_name in emb_grad_global:
                emb_grad_global[feature_name] /= self.size

            # Update neural network weights
            for i in range(self.num_layers):
                self.weights[i] -= lr * dW_global[i]
                self.biases[i] -= lr * db_global[i]

            # Update embedding weights
            for feature_name, embedding_layer in self.embedding_layers.items():
                embedding_layer.weights -= lr * emb_grad_global[feature_name]

            avg_loss = total_loss_sum / total_samples
        else:
            avg_loss = 0.0

        return avg_loss

    def parallel_evaluate(self, X_numerical_local, X_categorical_local, y_local):
        """Compute RMSE across all processes"""
        if X_numerical_local.shape[0] > 0:
            predictions = self.forward(X_numerical_local, X_categorical_local)
            local_se = np.sum((predictions.flatten() - y_local)**2)
            local_count = len(y_local)
        else:
            local_se = 0.0
            local_count = 0

        # Aggregate across processes
        local_stats = np.array([local_se, local_count], dtype=np.float64)
        global_stats = np.zeros(2, dtype=np.float64)
        self.comm.Allreduce(local_stats, global_stats, op=MPI.SUM)

        total_se = global_stats[0]
        total_count = int(global_stats[1])

        if total_count > 0:
            rmse = np.sqrt(total_se / total_count)
        else:
            rmse = 0.0

        return rmse

    def save_weights(self):
        """Save current weights and embeddings as best"""
        self.best_weights = {
            'weights': [w.copy() for w in self.weights],
            'biases': [b.copy() for b in self.biases]
        }
        self.best_embeddings = {
            name: emb.weights.copy() 
            for name, emb in self.embedding_layers.items()
        }

    def restore_best_weights(self):
        """Restore best weights and embeddings"""
        if self.best_weights:
            self.weights = [w.copy() for w in self.best_weights['weights']]
            self.biases = [b.copy() for b in self.best_weights['biases']]
        
        if self.best_embeddings:
            for name, embedding_layer in self.embedding_layers.items():
                embedding_layer.weights = self.best_embeddings[name].copy()

    def train(self, X_numerical_train_local, X_categorical_train_local, y_train_local,
              X_numerical_test_local, X_categorical_test_local, y_test_local,
              max_epochs=20, batch_size=1024, lr=0.001, patience=10,
              min_delta=1e-6, verbose=True):
        """Train the neural network with early stopping"""
        if self.rank == 0 and verbose:
            print(f"\nStarting training:")
            print(f"  Architecture: {self.layer_sizes}")
            print(f"  Numerical features: {self.numerical_features_size}")
            print(f"  Categorical features: {list(self.categorical_config.keys())}")
            print(f"  Total embedding size: {self.total_embedding_size}")
            print(f"  Max epochs: {max_epochs}")
            print(f"  Batch size: {batch_size:,}")
            print(f"  Learning rate: {lr}")
            print(f"  Activation: {self.activation}")
            print(f"  Processes: {self.size}")
            print(f"  Patience: {patience} epochs")

            # Data distribution info
            total_train = self.comm.reduce(X_numerical_train_local.shape[0], op=MPI.SUM, root=0)
            total_test = self.comm.reduce(X_numerical_test_local.shape[0], op=MPI.SUM, root=0)
            print(f"  Training samples: {total_train:,}")
            print(f"  Test samples: {total_test:,}")
        else:
            self.comm.reduce(X_numerical_train_local.shape[0], op=MPI.SUM, root=0)
            self.comm.reduce(X_numerical_test_local.shape[0], op=MPI.SUM, root=0)

        start_time = time.time()
        converged = False
        actual_epochs = 0

        for epoch in range(max_epochs):
            # Training step
            loss = self.train_step(
                X_numerical_train_local, X_categorical_train_local, y_train_local,
                batch_size, lr
            )

            # Evaluate
            train_rmse = self.parallel_evaluate(
                X_numerical_train_local, X_categorical_train_local, y_train_local
            )
            test_rmse = self.parallel_evaluate(
                X_numerical_test_local, X_categorical_test_local, y_test_local
            )

            # Store history
            self.history['loss'].append(loss)
            self.history['train_rmse'].append(train_rmse)
            self.history['test_rmse'].append(test_rmse)
            self.history['epochs'].append(epoch + 1)

            # Check for improvement
            if test_rmse < self.best_test_rmse - min_delta:
                self.best_test_rmse = test_rmse
                self.patience_counter = 0
                self.save_weights()
                improvement_str = " *"
            else:
                self.patience_counter += 1
                improvement_str = ""

            # Print progress
            if self.rank == 0 and verbose:
                print(f"Epoch {epoch+1:3d}/{max_epochs}: "
                      f"Loss={loss:.6f}, "
                      f"Train RMSE={train_rmse:.6f}, "
                      f"Test RMSE={test_rmse:.6f}{improvement_str}")

            # Early stopping
            if self.patience_counter >= patience:
                if self.rank == 0 and verbose:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    print(f"Best test RMSE: {self.best_test_rmse:.6f}")
                converged = True
                actual_epochs = epoch + 1
                break

            actual_epochs = epoch + 1

        # Restore best weights
        if self.best_weights:
            self.restore_best_weights()
            if self.rank == 0 and verbose:
                print("Restored best weights and embeddings")

        training_time = time.time() - start_time

        if self.rank == 0 and verbose:
            print(f"\nTraining completed in {training_time:.2f} seconds")
            print(f"Actual epochs: {actual_epochs}")
            print(f"Converged: {converged}")

        return training_time, actual_epochs, converged

def load_processed_data(file_path):
    """Load the processed taxi data with embeddings setup"""
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"Loading processed data from {file_path}...")
    
    df = pd.read_csv(file_path)
    
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"Loaded {len(df):,} samples with {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
    
    # Create target variable (trip_duration as proxy for fare)
    # You can replace this with actual fare if available
    if 'trip_duration' in df.columns:
        target_col = 'trip_duration'
    else:
        # Create a synthetic target based on trip_distance and other factors
        # This is just an example - replace with your actual target variable
        df['synthetic_fare'] = df['trip_distance'] * 2.5 + df['extra'] + np.random.normal(0, 0.5, len(df))
        target_col = 'synthetic_fare'
    
    # Define categorical and numerical features based on your preprocessing
    categorical_features = ['RatecodeID', 'PULocationID', 'DOLocationID', 'payment_type', 'pickup_hour', 'pickup_weekday']
    numerical_features = ['passenger_count', 'trip_distance', 'extra']
    
    # Add trip_duration to numerical if it's not the target
    if target_col != 'trip_duration' and 'trip_duration' in df.columns:
        numerical_features.append('trip_duration')
    
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"Categorical features: {categorical_features}")
        print(f"Numerical features: {numerical_features}")
        print(f"Target: {target_col}")
    
    # Prepare numerical features
    X_numerical = df[numerical_features].values.astype(np.float64)
    
    # Prepare categorical features and create embedding configuration
    X_categorical = {}
    categorical_config = {}
    
    for feature in categorical_features:
        if feature in df.columns:
            X_categorical[feature] = df[feature].values.astype(int)
            
            # Calculate embedding dimensions and number of categories
            num_categories = int(df[feature].max() + 1)
            embedding_dim = min(50, max(4, num_categories // 2))  # Rule of thumb for embedding size
            
            categorical_config[feature] = {
                'num_categories': num_categories,
                'embedding_dim': embedding_dim
            }
            
            if MPI.COMM_WORLD.Get_rank() == 0:
                print(f"  {feature}: {num_categories} categories -> {embedding_dim}D embedding")
    
    # Target variable
    y = df[target_col].values.astype(np.float64)
    
    # Remove any rows with invalid target values
    valid_indices = ~(np.isnan(y) | np.isinf(y))
    X_numerical = X_numerical[valid_indices]
    for feature in X_categorical:
        X_categorical[feature] = X_categorical[feature][valid_indices]
    y = y[valid_indices]
    
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"After filtering invalid targets: {len(y):,} samples")
        print(f"Target range: {y.min():.2f} - {y.max():.2f}")
        print(f"Target mean: {y.mean():.2f}")
    
    # Train/test split (70/30)
    n = len(y)
    n_train = int(0.7 * n)
    
    np.random.seed(42)
    indices = np.random.permutation(n)
    train_idx, test_idx = indices[:n_train], indices[n_train:]
    
    # Split data
    X_numerical_train = X_numerical[train_idx]
    X_numerical_test = X_numerical[test_idx]
    
    X_categorical_train = {feature: X_categorical[feature][train_idx] for feature in X_categorical}
    X_categorical_test = {feature: X_categorical[feature][test_idx] for feature in X_categorical}
    
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    return (X_numerical_train, X_categorical_train, y_train,
            X_numerical_test, X_categorical_test, y_test,
            X_numerical.shape[1], categorical_config)

def distribute_data(comm, X_numerical_train, X_categorical_train, y_train,
                   X_numerical_test, X_categorical_test, y_test):
    """Distribute data across MPI processes"""
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        n_train = len(X_numerical_train)
        n_test = len(X_numerical_test)

        # Split training data
        train_chunks_X_num = []
        train_chunks_X_cat = []
        train_chunks_y = []
        chunk_size = n_train // size
        remainder = n_train % size

        start = 0
        for i in range(size):
            end = start + chunk_size + (1 if i < remainder else 0)
            train_chunks_X_num.append(X_numerical_train[start:end].copy())
            train_chunks_X_cat.append({
                feature: X_categorical_train[feature][start:end].copy() 
                for feature in X_categorical_train
            })
            train_chunks_y.append(y_train[start:end].copy())
            start = end

        # Split test data
        test_chunks_X_num = []
        test_chunks_X_cat = []
        test_chunks_y = []
        chunk_size = n_test // size
        remainder = n_test % size

        start = 0
        for i in range(size):
            end = start + chunk_size + (1 if i < remainder else 0)
            test_chunks_X_num.append(X_numerical_test[start:end].copy())
            test_chunks_X_cat.append({
                feature: X_categorical_test[feature][start:end].copy() 
                for feature in X_categorical_test
            })
            test_chunks_y.append(y_test[start:end].copy())
            start = end
    else:
        train_chunks_X_num = None
        train_chunks_X_cat = None
        train_chunks_y = None
        test_chunks_X_num = None
        test_chunks_X_cat = None
        test_chunks_y = None

    X_numerical_train_local = comm.scatter(train_chunks_X_num, root=0)
    X_categorical_train_local = comm.scatter(train_chunks_X_cat, root=0)
    y_train_local = comm.scatter(train_chunks_y, root=0)
    X_numerical_test_local = comm.scatter(test_chunks_X_num, root=0)
    X_categorical_test_local = comm.scatter(test_chunks_X_cat, root=0)
    y_test_local = comm.scatter(test_chunks_y, root=0)

    return (X_numerical_train_local, X_categorical_train_local, y_train_local,
            X_numerical_test_local, X_categorical_test_local, y_test_local)

def save_training_plot(history, config, output_dir, arch_name):
    """Save training history plot"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['epochs'], history['loss'], 'b-', linewidth=2, label='Training Loss R(θk)')
    ax1.set_xlabel('Epoch k')
    ax1.set_ylabel('Loss R(θk)')
    ax1.set_title(f'Training Loss vs Epoch - {arch_name}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # RMSE plot
    ax2.plot(history['epochs'], history['train_rmse'], 'b-', linewidth=2, label='Training RMSE')
    ax2.plot(history['epochs'], history['test_rmse'], 'r-', linewidth=2, label='Test RMSE')
    ax2.set_xlabel('Epoch k')
    ax2.set_ylabel('RMSE')
    ax2.set_title(f'RMSE vs Epoch - {arch_name}')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save plot
    layers_str = "-".join(map(str, config['hidden_layers']))
    filename = f"training_history_{config['activation']}_{arch_name}_layers{layers_str}_b{config['batch_size']}_p{config['num_processes']}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

def main():
    parser = argparse.ArgumentParser(description='Auto-Compare Taxi Fare Prediction Architectures with MPI and Embeddings')
    parser.add_argument('--data', type=str, default='processed_data.csv',
                       help='Path to processed data file')
    parser.add_argument('--max-epochs', type=int, default=20,
                       help='Maximum number of epochs')
    parser.add_argument('--batch-size', type=int, default=1024,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--activation', choices=['relu', 'sigmoid', 'tanh'],
                       default='relu', help='Activation function')
    parser.add_argument('--patience', type=int, default=10,
                       help='Patience for early stopping')
    parser.add_argument('--min-delta', type=float, default=1e-6,
                       help='Minimum improvement for early stopping')

    args = parser.parse_args()

    # Define architectures to test
    architectures_to_test = {
        'arch_512_256': [512, 256],
        'arch_256_128': [256, 128]
    }

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("="*60)
        print("Auto-Compare NYC Taxi Fare Prediction - Neural Network Architectures")
        print("="*60)
        print(f"MPI Configuration: {size} processes")
        print("Testing architectures:")
        print("  - Architecture 1: 512,256 (512 → 256 → output)")
        print("  - Architecture 2: 256,128 (256 → 128 → output)")

    # Load data
    if rank == 0:
        (X_numerical_train, X_categorical_train, y_train,
         X_numerical_test, X_categorical_test, y_test,
         n_numerical_features, categorical_config) = load_processed_data(args.data)
    else:
        X_numerical_train = X_categorical_train = y_train = None
        X_numerical_test = X_categorical_test = y_test = None
        n_numerical_features = None
        categorical_config = None

    # Broadcast configuration
    n_numerical_features = comm.bcast(n_numerical_features, root=0)
    categorical_config = comm.bcast(categorical_config, root=0)

    # Distribute data
    (X_numerical_train_local, X_categorical_train_local, y_train_local,
     X_numerical_test_local, X_categorical_test_local, y_test_local) = distribute_data(
        comm, X_numerical_train, X_categorical_train, y_train,
        X_numerical_test, X_categorical_test, y_test
    )

    # Test both architectures and compare
    best_architecture = None
    best_test_rmse = float('inf')
    best_result = None
    all_results = []

    if rank == 0:
        print("\n" + "="*60)
        print("TESTING BOTH ARCHITECTURES")
        print("="*60)
        print("| Architecture | Train RMSE | Test RMSE | Epochs | Time (s) | Converged |")
        print("|--------------|------------|-----------|--------|----------|-----------|")

    for arch_name, hidden_layers in architectures_to_test.items():
        if rank == 0:
            print(f"\nTesting {arch_name}: {hidden_layers}")
        
        # Create and train model
        model = MultiLayerTaxiNN(
            numerical_features_size=n_numerical_features,
            categorical_config=categorical_config,
            hidden_layers=hidden_layers,
            activation=args.activation
        )

        training_time, actual_epochs, converged = model.train(
            X_numerical_train_local, X_categorical_train_local, y_train_local,
            X_numerical_test_local, X_categorical_test_local, y_test_local,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            patience=args.patience,
            min_delta=args.min_delta,
            verbose=False  # Reduce verbose output during comparison
        )
        
        # Get final metrics
        final_train_rmse = model.history['train_rmse'][-1] if model.history['train_rmse'] else 0
        final_test_rmse = model.history['test_rmse'][-1] if model.history['test_rmse'] else 0
        
        # Store results
        result = {
            'architecture_name': arch_name,
            'hidden_layers': hidden_layers,
            'activation': args.activation,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'max_epochs': args.max_epochs,
            'actual_epochs': actual_epochs,
            'converged': converged,
            'num_processes': size,
            'train_rmse': final_train_rmse,
            'test_rmse': final_test_rmse,
            'training_time': training_time,
            'categorical_config': categorical_config,
            'n_numerical_features': n_numerical_features,
            'history': model.history
        }
        all_results.append(result)
        
        if rank == 0:
            status = "Yes" if converged else "No"
            arch_display = str(hidden_layers).replace(' ', '')
            print(f"| {arch_display:12} | {final_train_rmse:10.6f} | {final_test_rmse:9.6f} | {actual_epochs:6d} | {training_time:8.2f} | {status:9} |")
        
        # Check if this is the best architecture
        if final_test_rmse < best_test_rmse:
            best_test_rmse = final_test_rmse
            best_architecture = arch_name
            best_result = result

    if rank == 0:
        print("\n" + "="*60)
        print("ARCHITECTURE COMPARISON RESULTS")
        print("="*60)
        print(f"Best Architecture: {best_result['architecture_name']} - {best_result['hidden_layers']}")
        print(f"Best Test RMSE: {best_result['test_rmse']:.6f}")
        print(f"Training Time: {best_result['training_time']:.2f} seconds")
        print(f"Converged: {best_result['converged']}")
        
        # Show detailed comparison
        arch1_result = next(r for r in all_results if r['architecture_name'] == 'arch_512_256')
        arch2_result = next(r for r in all_results if r['architecture_name'] == 'arch_256_128')
        
        rmse_diff = abs(arch1_result['test_rmse'] - arch2_result['test_rmse'])
        time_diff = abs(arch1_result['training_time'] - arch2_result['training_time'])
        
        print(f"\nDetailed Comparison:")
        print(f"  Architecture 512,256:")
        print(f"    Test RMSE: {arch1_result['test_rmse']:.6f}")
        print(f"    Training Time: {arch1_result['training_time']:.2f}s")
        print(f"    Epochs: {arch1_result['actual_epochs']}")
        print(f"    Converged: {arch1_result['converged']}")
        
        print(f"  Architecture 256,128:")
        print(f"    Test RMSE: {arch2_result['test_rmse']:.6f}")
        print(f"    Training Time: {arch2_result['training_time']:.2f}s")
        print(f"    Epochs: {arch2_result['actual_epochs']}")
        print(f"    Converged: {arch2_result['converged']}")
        
        print(f"\nPerformance Differences:")
        print(f"  RMSE Difference: {rmse_diff:.6f}")
        print(f"  Time Difference: {time_diff:.2f}s")
        
        # Determine winner
        if arch1_result['test_rmse'] < arch2_result['test_rmse']:
            improvement = arch2_result['test_rmse'] - arch1_result['test_rmse']
            pct_improvement = (improvement / arch2_result['test_rmse']) * 100
            print(f"\nWinner: 512,256 architecture")
            print(f"  Improvement: {improvement:.6f} RMSE ({pct_improvement:.2f}% better)")
        else:
            improvement = arch1_result['test_rmse'] - arch2_result['test_rmse']
            pct_improvement = (improvement / arch1_result['test_rmse']) * 100
            print(f"\nWinner: 256,128 architecture")
            print(f"  Improvement: {improvement:.6f} RMSE ({pct_improvement:.2f}% better)")

        # Embedding information
        print(f"\nEmbedding Configuration:")
        for feature_name, config in categorical_config.items():
            print(f"  {feature_name}: {config['num_categories']} categories → {config['embedding_dim']}D")

        # Save results for both architectures
        os.makedirs('results', exist_ok=True)
        os.makedirs('results/plots', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save individual results
        for result in all_results:
            layers_str = "-".join(map(str, result['hidden_layers']))
            result_file = f'results/taxi_result_{result["activation"]}_{result["architecture_name"]}_layers{layers_str}_b{result["batch_size"]}_p{size}_{timestamp}.json'
            
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Save training plot
            plot_file = save_training_plot(result['history'], result, 'results/plots', result['architecture_name'])
            print(f"Saved: {result_file}")
            print(f"Plot: {plot_file}")
        
        # Save comparison summary
        comparison_summary = {
            'timestamp': timestamp,
            'experiment_config': {
                'activation': args.activation,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'max_epochs': args.max_epochs,
                'num_processes': size
            },
            'data_config': {
                'n_numerical_features': n_numerical_features,
                'categorical_config': categorical_config
            },
            'best_architecture': best_result['architecture_name'],
            'best_test_rmse': best_result['test_rmse'],
            'architectures_tested': {
                'arch_512_256': {
                    'test_rmse': arch1_result['test_rmse'],
                    'training_time': arch1_result['training_time'],
                    'converged': arch1_result['converged'],
                    'epochs': arch1_result['actual_epochs']
                },
                'arch_256_128': {
                    'test_rmse': arch2_result['test_rmse'],
                    'training_time': arch2_result['training_time'],
                    'converged': arch2_result['converged'],
                    'epochs': arch2_result['actual_epochs']
                }
            },
            'performance_difference': {
                'rmse_diff': rmse_diff,
                'time_diff': time_diff,
                'winner': best_result['architecture_name']
            },
            'all_results': all_results
        }
        
        comparison_file = f'results/architecture_comparison_{args.activation}_b{args.batch_size}_p{size}_{timestamp}.json'
        with open(comparison_file, 'w') as f:
            json.dump(comparison_summary, f, indent=2)
        
        print(f"\nComparison summary saved: {comparison_file}")
        
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETED")
        print("="*60)
        print("Files generated:")
        print(f"  - Individual results: results/taxi_result_*.json")
        print(f"  - Training plots: results/plots/*.png")
        print(f"  - Comparison summary: {comparison_file}")
        print("\nRecommendation:")
        print(f"  Use {best_result['architecture_name']} ({best_result['hidden_layers']}) for best performance")
        print(f"  Expected Test RMSE: {best_result['test_rmse']:.6f}")

if __name__ == "__main__":
    main()