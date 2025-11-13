"""
Block-Recurrent Transformer Architecture. Example, not necessarily how we will do things.

This will use *real world* weather data to predict the weather. 
This file will be how we would do things *without* PyTorch, so this is like the "from scratch" method.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import requests
from datetime import datetime, timedelta
import time
import os

from data_fetch.real_weather_data import *


class RecurrentTransformersBlock:
    """
    Single block of the recurrent transformer
    """
    def __init__(self, hidden_dim, state_dim, num_heads=4):
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Token path (Transformer) weights
        self.token_qkv = self._init_weights((hidden_dim, 3 * hidden_dim))
        self.token_cross_q = self._init_weights((hidden_dim, hidden_dim))
        self.token_cross_kv = self._init_weights((state_dim, 2 * hidden_dim))
        self.token_ffn_w1 = self._init_weights((hidden_dim, 2 * hidden_dim))
        self.token_ffn_w2 = self._init_weights((2 * hidden_dim, hidden_dim))

        # State path (Recurrent) weights
        self.state_qkv = self._init_weights((state_dim, 3 * state_dim))
        self.state_cross_q = self._init_weights((state_dim, state_dim))
        self.state_cross_kv = self._init_weights((hidden_dim, 2 * state_dim))
        self.state_gate_w = self._init_weights((state_dim, state_dim))
        self.state_ffn_w1 = self._init_weights((state_dim, 2 * state_dim))
        self.state_ffn_w2 = self._init_weights((2 * state_dim, state_dim))

    def _init_weights(self, shape):
        return np.random.randn(*shape) * 0.02

    def attention(self, q, k, v):
        """Scaled dot-product attention"""
        d_k = q.shape[-1]

        # For (batch, seq_len, dim), swap last two dimensions
        k_transposed = k.transpose(0, 2, 1)

        scores = np.matmul(q, k_transposed) / np.sqrt(d_k)
        attn_weights = self.softmax(scores)
        return np.matmul(attn_weights, v)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def layer_norm(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True) + 1e-6
        return (x - mean) / std
    
    def forward(self, tokens, state):
        """
        Forward pass through the block
        tokens: (batch, seq_len, hidden_dim)
        state: (batch, state_len, state_dim)
        """
        batch_size = tokens.shape[0]

        # === TOKEN PATH (Transformer) ===
        # Self-attention over tokens 
        token_qkv = np.matmul(tokens, self.token_qkv)
        token_q, token_k, token_v = np.split(token_qkv, 3, axis=-1)
        tokens_attn = self.attention(token_q, token_k, token_v)
        tokens = self.layer_norm(tokens + tokens_attn)

        # Cross-attention: tokens attend to state
        token_cross_q = np.matmul(tokens, self.token_cross_q)
        state_kv = np.matmul(state, self.token_cross_kv)
        state_k, state_v = np.split(state_kv, 2, axis=-1)
        tokens_cross = self.attention(token_cross_q, state_k, state_v)
        tokens = self.layer_norm(tokens + tokens_cross)

        # Feed-forward network
        token_ffn = np.matmul(tokens, self.token_ffn_w1)
        token_ffn = np.maximum(0, token_ffn)  # ReLU
        token_ffn = np.matmul(token_ffn, self.token_ffn_w2)
        tokens = self.layer_norm(tokens + token_ffn)

        # === STATE PATH (Recurrent) ===
        # Self-attention over states
        state_qkv = np.matmul(state, self.state_qkv)
        state_q, state_k, state_v = np.split(state_qkv, 3, axis=-1)
        state_attn = self.attention(state_q, state_k, state_v)
        state = self.layer_norm(state + state_attn)

        # Cross-attention: state attends to tokens
        state_cross_q = np.matmul(state, self.state_cross_q)
        token_kv = np.matmul(tokens, self.state_cross_kv)
        token_k, token_v = np.split(token_kv, 2, axis=-1)
        state_cross = self.attention(state_cross_q, token_k, token_v)

        # Gating (LSTM-like)
        gate = 1 / (1 + np.exp(-np.matmul(state, self.state_gate_w)))   # Sigmoid
        state = state * gate + state_cross * (1 - gate)
        state = self.layer_norm(state)

        # Feed-forward network
        state_ffn = np.matmul(state, self.state_ffn_w1)
        state_ffn = np.maximum(0, state_ffn)  # ReLU
        state_ffn = np.matmul(state_ffn, self.state_ffn_w2)
        state = self.layer_norm(state + state_ffn)

        return tokens, state


class RecurrentTransformer:
    """
    Full Recurrent Transformer model
    """
    def __init__(self, input_dim, hidden_dim, state_dim, num_blocks, block_size):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.num_blocks = num_blocks
        self.block_size = block_size

        # Input embedding
        self.input_embed = np.random.randn(input_dim, hidden_dim) * 0.02

        # Positional encoding
        self.pos_encoding = self._get_positional_encoding(100, hidden_dim)

        # Blocks
        self.blocks = [
            RecurrentTransformersBlock(hidden_dim, state_dim)
            for _ in range(num_blocks)
        ]

        # Output projection
        self.output_proj = np.random.randn(hidden_dim, input_dim) * 0.02

    def _get_positional_encoding(self, seq_len, d_model):
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pos_encoding = np.zeros((seq_len, d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        return pos_encoding

    def forward(self, x, verbose=False):
        """
        Forward pass
        x: (batch, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Embed input
        x = np.matmul(x, self.input_embed)
        x = x + self.pos_encoding[:seq_len]

        if verbose:
            print(f"Input embedded: {x.shape}")

        # Split into blocks
        num_full_blocks = seq_len // self.block_size
        blocks_data = []
        for i in range(num_full_blocks):
            start = i * self.block_size
            end = start + self.block_size
            blocks_data.append(x[:, start:end, :])

        if verbose:
            print(f"Split into {len(blocks_data)} blocks of size {self.block_size}")

        # Initialize state
        state = np.zeros((batch_size, 1, self.state_dim))

        # Process each block
        outputs = []
        for i, block_data in enumerate(blocks_data):
            if i >= self.num_blocks:
                break

            block_data, state = self.blocks[i].forward(block_data, state)
            outputs.append(block_data)

            if verbose:
                print(f"Block {i+1} processed: tokens {block_data.shape}, state {state.shape}")

        # Concatenate outputs
        output = np.concatenate(outputs, axis=1)

        # Project to output
        output = np.matmul(output, self.output_proj)

        if verbose:
            print(f"Final output: {output.shape}")

        return output, state


def train_step(model, X_batch, y_batch, learning_rate=0.001):
    """
    Simple training step (gradient descent approximation)
    In practice, you'd use proper backpropagation
    """

    # Forward pass
    pred, _ = model.forward(X_batch)

    # Only predict the last timestep
    pred_last = pred[:, -1, :]
    y_last = y_batch[:, -1, :]

    # Loss (MSE)
    loss = np.mean((pred_last - y_last) ** 2)

    # Simple gradient approximation (finite differences)
    # In practice, use autograd/PyTorch/JAX
    epsilon = 1e-7
    for block in model.blocks:
        for attr in dir(block):
            if attr.endswith('_w') or attr.endswith('_w1') or attr.endswith('_w2') or 'qkv' in attr or 'kv' in attr or attr.endswith('_q'):
                weights = getattr(block, attr)
                grad = np.zeros_like(weights)

                # Sample a few random positions for gradient estimation 
                for _ in range(min(10, weights.size)):
                    i, j = np.random.randint(0, weights.shape[0]), np.random.randint(0, weights.shape[1])

                    # Perturb weight
                    original = weights[i, j]
                    weights[i, j] = original + epsilon
                    pred_plus, _ = model.forward(X_batch)
                    loss_plus = np.mean((pred_plus[:, -1, :] - y_last) ** 2)

                    weights[i, j] = original - epsilon
                    pred_minus, _ = model.forward(X_batch)
                    loss_minus = np.mean((pred_minus[:, -1, :] - y_last) ** 2)

                    weights[i, j] = original
                    grad[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

                # Update
                weights -= learning_rate * grad

    return loss


def main():
    print(f"Recurrent Transformer (\"from scratch\") - Real Weather Data")

    cities = [
        (40.7128, -74.0060, "New York"),
        (51.5074, -0.1278, "London"),
        (35.6762, 139.6503, "Tokyo"),
        (34.0522, -118.2437, "Los Angeles"),
        (-33.8688, 151.2093, "Sydney"),
        (38.8121, -77.6364, "Haymarket"),
        (25.0330, 121.5654, "Taipei"),
        (13.0843, 80.2705, "Chennai")
    ]

    print(f"\n1. Fetching Real World Weather data from {len(cities)} cities.")

    all_temp_data = []

    for lat, lon, name in cities:
        weather_data = fetch_real_weather_data(lat, lon, days_back=365, location_name=name)
        if weather_data is not None:
            all_temp_data.append(weather_data["temperature"])
        time.sleep(0.5)
    
    if not all_temp_data:
        print("Failed to fetch weather data. Exiting.")
        return
    
    print(f"\nTotal data collected from {len(all_temp_data)} cities")

    # Prepare sequences BEFORE concatenating
    print("\n2. Preparing sequences...")
    seq_length = 48  # 48 hours input
    pred_length = 24  # 24 hours prediction

    all_X, all_y = [], []
    for temp_data in all_temp_data:
        X_seq, y_seq = prepare_sequences(temp_data, seq_length, pred_length)
        all_X.append(X_seq)
        all_y.append(y_seq)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    # Add feature dimension
    X = X.reshape(X.shape[0], X.shape[1], 1)
    y = y.reshape(y.shape[0], y.shape[1], 1)

    print(f"   Created {X.shape[0]} sequences")
    print(f"   Input shape: {X.shape}")
    print(f"   Target shape: {y.shape}")

    # Normalize
    scaler = StandardScaler()
    X_flat = X.reshape(-1, 1)
    y_flat = y.reshape(-1, 1)

    scaler.fit(X_flat)
    X = scaler.transform(X_flat).reshape(X.shape)
    y = scaler.transform(y_flat).reshape(y.shape)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   Training set: {X_train.shape}, Test set: {X_test.shape}")

    # Initialize model
    print("\n2. Initializing Recurrent Transformer...")
    model = RecurrentTransformer(
        input_dim=1,   # Single feature (temperature)
        hidden_dim=32, # Hidden dimension
        state_dim=16,  # State dimension
        num_blocks=4,  # Number of blocks
        block_size=12  # Tokens per block
    )
    print(f"   Model: {model.num_blocks} blocks, hidden_dim={model.hidden_dim}, state_dim={model.state_dim}")

    # Run forward pass on one example
    print("\n3. Running forward pass (verbose)...")
    sample_input = X_train[0:1]
    output, state = model.forward(sample_input, verbose=True)

    # Make prediction on test set
    print("\n4. Making predictions on test set...")
    predictions, _ = model.forward(X_test[:5])

    print("\n5. Sample predictions vs actual:")
    for i in range(3):
        pred_last = predictions[i, -1, 0]
        actual_last = y_test[i, -1, 0]
        print(f"   Sample {i+1}: Predicted={pred_last:.3f}, Actual={actual_last:.3f}")

    # Visualize
    print("\n6. Creating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Recurrent Transformer - Weather Forecasting', fontsize=16, fontweight='bold')

    # Plot 1: Sample sequences (use raw temperature data from first city)
    axes[0, 0].set_title('Sample Training Sequences (First 5 days)')
    for i in range(min(5, len(all_temp_data))):
        # Plot first 120 hours (5 days) for visibility
        axes[0, 0].plot(all_temp_data[i][:120], alpha=0.7, label=f'{cities[i][2]}')
    axes[0, 0].set_xlabel('Time Step (hours)')
    axes[0, 0].set_ylabel('Temperature (°C)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Input vs Target
    axes[0, 1].set_title('Input Sequence vs Target Sequence')
    sample_idx = 0
    # Note: X_train has 48 timesteps, y_train has 24 timesteps
    axes[0, 1].plot(range(48), X_train[sample_idx].flatten(), 'b-', label='Input (48 hours)', linewidth=2)
    axes[0, 1].plot(range(48, 72), y_train[sample_idx].flatten(), 'r-', label='Target (next 24 hours)', linewidth=2)
    axes[0, 1].axvline(x=47, color='gray', linestyle='--', alpha=0.5, label='Prediction start')
    axes[0, 1].set_xlabel('Time Step (hours)')
    axes[0, 1].set_ylabel('Normalized Temperature')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Full Sequence - Input + Prediction vs Actual
    axes[1, 0].set_title('Full Sequence: Input + Prediction')
    for i in range(3):
        # Full prediction output (48 hours)
        pred_full = predictions[i, :, 0]
        
        # Show input part (first 48 hours - from X_test)
        input_part = X_test[i, :, 0]
        
        # Show target (next 24 hours)
        actual = y_test[i, :, 0]
        
        # Plot input
        axes[1, 0].plot(range(48), input_part, ':', alpha=0.4, color=f'C{i}', label=f'Input {i+1}')
        # Plot prediction (last 24 of output)
        axes[1, 0].plot(range(48, 72), pred_full[-24:], 's--', alpha=0.6, color=f'C{i}', label=f'Pred {i+1}')
        # Plot actual
        axes[1, 0].plot(range(48, 72), actual, 'o-', alpha=0.6, color=f'C{i}', label=f'Actual {i+1}')
        
    axes[1, 0].axvline(x=47, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Time Step (hours)')
    axes[1, 0].set_ylabel('Normalized Temperature')
    axes[1, 0].legend(fontsize=7, ncol=3)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Architecture diagram
    axes[1, 1].axis('off')
    architecture_text = """
    RECURRENT TRANSFORMER ARCHITECTURE

    Input: (batch, 48, 1)
    ↓
    Embedding: → (batch, 48, 32)
    ↓
    Split into Blocks: 4 blocks × 12 tokens
    ↓
    ┌─────────────────────────────────┐
    │  BLOCK 1 (tokens 0-11)         │
    │  ┌──────────┐  ┌──────────┐    │
    │  │  TOKEN   │  │  STATE   │    │
    │  │   PATH   │←→│   PATH   │    │
    │  │(Transform)  │(Recurrent)│    │
    │  └──────────┘  └──────────┘    │
    │  State S₀ → State S₁            │
    └─────────────────────────────────┘
    ↓ (State flows to next block)
    ┌─────────────────────────────────┐
    │  BLOCK 2 (tokens 12-23)        │
    │  State S₁ → State S₂            │
    └─────────────────────────────────┘
    ↓
    BLOCK 3, BLOCK 4...
    ↓
    Output Projection: → (batch, 24, 1)
    ↓
    Prediction: Next 24 hours

    Key Features:
    - Token Path: Self-Attn → Cross-Attn(State)
    - State Path: Self-Attn → Cross-Attn(Token) → Gate
    - Recurrent state carries memory between blocks
    """
    axes[1, 1].text(0.1, 0.5, architecture_text, fontsize=9, 
                    verticalalignment='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig('recurrent_transformer_results.png', dpi=150, bbox_inches='tight')
    print("   Saved visualization to 'recurrent_transformer_results.png'")

    print("\nNote: This model uses random weights (not trained).")
    print("For real performance, you'd need to implement backpropagation")
    print("and train for many epochs. Consider using PyTorch or JAX.")


if __name__ == "__main__":
    main()
