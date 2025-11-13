import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import requests
from datetime import datetime, timedelta
import time

from data_fetch.real_weather_data import *

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# def fetch_real_weather_data(latitude=40.7128, longitude=-74.0060, 
#                            days_back=365, location_name="New York"):
#     """
#     Fetch real historical weather data from Open-Meteo API
#     Free API, no key required!
    
#     Args:
#         latitude: Location latitude
#         longitude: Location longitude
#         days_back: Number of days of historical data to fetch
#         location_name: Name of location for display
#     """
#     print(f"\nFetching real weather data for {location_name}...")
#     print(f"  Coordinates: ({latitude}, {longitude})")
#     print(f"  Time range: Last {days_back} days")
    
#     # Calculate date range
#     end_date = datetime.now().date()
#     start_date = end_date - timedelta(days=days_back)
    
#     # Open-Meteo API endpoint for historical data
#     url = "https://archive-api.open-meteo.com/v1/archive"
    
#     params = {
#         "latitude": latitude,
#         "longitude": longitude,
#         "start_date": start_date.strftime("%Y-%m-%d"),
#         "end_date": end_date.strftime("%Y-%m-%d"),
#         "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
#         "timezone": "auto"
#     }
    
#     print(f"  API Request: {url}")
    
#     try:
#         response = requests.get(url, params=params, timeout=30)
#         response.raise_for_status()
#         data = response.json()
        
#         # Extract temperature data
#         if "hourly" in data and "temperature_2m" in data["hourly"]:
#             temperatures = np.array(data["hourly"]["temperature_2m"])
#             humidity = np.array(data["hourly"]["relative_humidity_2m"])
#             precipitation = np.array(data["hourly"]["precipitation"])
#             wind_speed = np.array(data["hourly"]["wind_speed_10m"])
#             timestamps = data["hourly"]["time"]
            
#             print(f"  ✓ Successfully fetched {len(temperatures)} hourly data points")
#             print(f"  Temperature range: {np.min(temperatures):.1f}°C to {np.max(temperatures):.1f}°C")
#             print(f"  Humidity range: {np.min(humidity):.1f}% to {np.max(humidity):.1f}%")
            
#             return {
#                 "temperature": temperatures,
#                 "humidity": humidity,
#                 "precipitation": precipitation,
#                 "wind_speed": wind_speed,
#                 "timestamps": timestamps,
#                 "location": location_name,
#                 "coords": (latitude, longitude)
#             }
#         else:
#             print("  ✗ Error: No data returned from API")
#             return None
            
#     except Exception as e:
#         print(f"  ✗ Error fetching data: {e}")
#         return None


# def prepare_sequences(data, seq_length=48, pred_length=24):
#     """
#     Prepare sequences for training
    
#     Args:
#         data: Weather data array
#         seq_length: Length of input sequence (hours)
#         pred_length: Length of prediction sequence (hours)
#     """
#     sequences = []
#     targets = []
    
#     for i in range(len(data) - seq_length - pred_length):
#         # Input: seq_length hours
#         seq = data[i:i + seq_length]
#         # Target: next pred_length hours
#         target = data[i + seq_length:i + seq_length + pred_length]
        
#         sequences.append(seq)
#         targets.append(target)
    
#     return np.array(sequences), np.array(targets)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections and split into heads
        Q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads and apply final linear
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(attn_output)


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class RecurrentTransformerBlock(nn.Module):
    """
    Single block of the recurrent transformer
    """
    def __init__(self, hidden_dim, state_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        
        # TOKEN PATH (Transformer)
        self.token_self_attn = MultiHeadAttention(hidden_dim, num_heads)
        self.token_cross_attn = MultiHeadAttention(hidden_dim, num_heads)
        self.token_ffn = FeedForward(hidden_dim, 2 * hidden_dim, dropout)
        
        self.token_norm1 = nn.LayerNorm(hidden_dim)
        self.token_norm2 = nn.LayerNorm(hidden_dim)
        self.token_norm3 = nn.LayerNorm(hidden_dim)
        
        # STATE PATH (Recurrent)
        self.state_self_attn = MultiHeadAttention(state_dim, num_heads)
        self.state_cross_attn = MultiHeadAttention(state_dim, num_heads)
        self.state_ffn = FeedForward(state_dim, 2 * state_dim, dropout)
        
        # Projection layers
        self.state_to_hidden = nn.Linear(state_dim, hidden_dim)
        self.hidden_to_state = nn.Linear(hidden_dim, state_dim)
        
        # Gating mechanism
        self.state_gate = nn.Linear(state_dim, state_dim)
        
        self.state_norm1 = nn.LayerNorm(state_dim)
        self.state_norm2 = nn.LayerNorm(state_dim)
        self.state_norm3 = nn.LayerNorm(state_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tokens, state):
        # TOKEN PATH (VERTICAL)
        token_attn = self.token_self_attn(tokens, tokens, tokens)
        tokens = self.token_norm1(tokens + self.dropout(token_attn))
        
        state_proj = self.state_to_hidden(state)
        token_cross = self.token_cross_attn(tokens, state_proj, state_proj)
        tokens = self.token_norm2(tokens + self.dropout(token_cross))
        
        token_ff = self.token_ffn(tokens)
        tokens = self.token_norm3(tokens + self.dropout(token_ff))
        
        # STATE PATH (HORIZONTAL)
        state_attn = self.state_self_attn(state, state, state)
        state_updated = self.state_norm1(state + self.dropout(state_attn))
        
        tokens_proj = self.hidden_to_state(tokens)
        state_cross = self.state_cross_attn(state_updated, tokens_proj, tokens_proj)
        
        gate = torch.sigmoid(self.state_gate(state_updated))
        state_gated = gate * state_updated + (1 - gate) * state_cross
        state_updated = self.state_norm2(state_gated)
        
        state_ff = self.state_ffn(state_updated)
        state_updated = self.state_norm3(state_updated + self.dropout(state_ff))
        
        return tokens, state_updated


class RecurrentTransformer(nn.Module):
    """Full Recurrent Transformer model"""
    def __init__(self, input_dim, hidden_dim, state_dim, num_blocks, 
                 block_size, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        
        self.input_embed = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = self._create_positional_encoding(1000, hidden_dim)
        
        self.blocks = nn.ModuleList([
            RecurrentTransformerBlock(hidden_dim, state_dim, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        self.init_state = nn.Parameter(torch.randn(1, 1, state_dim) * 0.02)
        
    def _create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
        
    def forward(self, x, return_state=False):
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        x = self.input_embed(x)
        x = x + self.pos_encoding[:, :seq_len, :].to(device)
        
        num_full_blocks = seq_len // self.block_size
        blocks_data = []
        for i in range(num_full_blocks):
            start = i * self.block_size
            end = start + self.block_size
            blocks_data.append(x[:, start:end, :])
        
        if seq_len % self.block_size != 0:
            blocks_data.append(x[:, num_full_blocks * self.block_size:, :])
        
        state = self.init_state.expand(batch_size, -1, -1)
        
        outputs = []
        for i, block_data in enumerate(blocks_data):
            block_idx = min(i, self.num_blocks - 1)
            block_output, state = self.blocks[block_idx](block_data, state)
            outputs.append(block_output)
        
        output = torch.cat(outputs, dim=1)
        output = self.output_proj(output)
        
        if return_state:
            return output, state
        return output


def train_model(model, train_loader, val_loader, num_epochs=30, lr=0.001, device='cpu'):
    """Train the recurrent transformer"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    print("\nTraining Recurrent Transformer on Real Weather Data...")
    print("=" * 70)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            output = model(X_batch)

            output_future = output[:, -24:, :]  # Take last 24 hours
            loss = criterion(output_future, y_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)

                output_future = output[:, -24:, :]
                loss = criterion(output_future, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    print("=" * 70)
    return train_losses, val_losses


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    print("\n" + "=" * 70)
    print("RECURRENT TRANSFORMER - REAL WEATHER DATA")
    print("=" * 70)
    
    # Fetch real weather data from multiple cities
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
    
    all_temp_data = []
    
    for lat, lon, name in cities:
        weather_data = fetch_real_weather_data(lat, lon, days_back=365, location_name=name)
        if weather_data is not None:
            all_temp_data.append(weather_data["temperature"])
        time.sleep(0.5)  # Be nice to the API
    
    if not all_temp_data:
        print("Failed to fetch weather data. Exiting.")
        return
    
    print(f"\n✓ Total data collected from {len(all_temp_data)} cities")
    
    # Prepare sequences
    print("\n2. Preparing sequences...")
    seq_length = 48  # 48 hours input (2 days)
    pred_length = 24  # 24 hours prediction (1 day)
    
    all_X, all_y = [], []
    for temp_data in all_temp_data:
        X, y = prepare_sequences(temp_data, seq_length, pred_length)
        all_X.append(X)
        all_y.append(y)
    
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    # Add feature dimension
    X = X.reshape(X.shape[0], X.shape[1], 1)
    y = y.reshape(y.shape[0], y.shape[1], 1)
    
    print(f"   Created {X.shape[0]} sequences")
    print(f"   Input shape: {X.shape} (samples, {seq_length} hours, 1 feature)")
    print(f"   Target shape: {y.shape} (samples, {pred_length} hours, 1 feature)")
    
    # Normalize
    scaler = StandardScaler()
    X_flat = X.reshape(-1, 1)
    y_flat = y.reshape(-1, 1)
    
    scaler.fit(X_flat)
    X_normalized = scaler.transform(X_flat).reshape(X.shape)
    y_normalized = scaler.transform(y_flat).reshape(y.shape)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y_normalized, test_size=0.15, random_state=42
    )
    
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"   Training: {len(X_train)} sequences, Test: {len(X_test)} sequences")
    
    # Initialize model
    print("\n3. Initializing Recurrent Transformer...")
    model = RecurrentTransformer(
        input_dim=1,
        hidden_dim=64,
        state_dim=32,
        num_blocks=4,
        block_size=12,  # 12 hours per block
        num_heads=4,
        dropout=0.1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Train
    train_losses, val_losses = train_model(
        model, train_loader, test_loader,
        num_epochs=30, lr=0.001, device=device
    )
    
    # Evaluate
    print("\n4. Evaluating on test set...")
    model.eval()
    all_predictions = []
    all_actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            output = model(X_batch)
            all_predictions.append(output.cpu())
            all_actuals.append(y_batch)
    
    predictions = torch.cat(all_predictions, dim=0)
    actuals = torch.cat(all_actuals, dim=0)
    
    # Inverse transform for real temperature values
    pred_flat = predictions.reshape(-1, 1).numpy()
    actual_flat = actuals.reshape(-1, 1).numpy()
    
    pred_real = scaler.inverse_transform(pred_flat).reshape(predictions.shape)
    actual_real = scaler.inverse_transform(actual_flat).reshape(actuals.shape)
    
    mse = np.mean((pred_real - actual_real) ** 2)
    mae = np.mean(np.abs(pred_real - actual_real))
    
    print(f"   Test MSE: {mse:.4f}°C²")
    print(f"   Test MAE: {mae:.4f}°C")
    
    # Visualize
    print("\n5. Creating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Recurrent Transformer - Real Weather Data Forecasting', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Training curves
    axes[0, 0].set_title('Training Progress')
    axes[0, 0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0, 0].plot(val_losses, label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Plot 2: Sample predictions
    axes[0, 1].set_title('24-Hour Temperature Forecast (Sample Predictions)')
    for i in range(min(3, len(pred_real))):
        hours = np.arange(24)
        axes[0, 1].plot(hours, actual_real[i, :, 0], 'o-', alpha=0.7, 
                       label=f'Actual {i+1}', linewidth=2)
        axes[0, 1].plot(hours, pred_real[i, :, 0], 's--', alpha=0.7, 
                       label=f'Predicted {i+1}', linewidth=2)
    axes[0, 1].set_xlabel('Hour')
    axes[0, 1].set_ylabel('Temperature (°C)')
    axes[0, 1].legend(fontsize=8, ncol=2)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Full sequence visualization
    axes[1, 0].set_title('Input + Prediction Sequence')
    sample_idx = 0
    # Denormalize input
    input_sample = scaler.inverse_transform(
        X_test[sample_idx].reshape(-1, 1)
    ).flatten()
    
    full_hours = np.arange(72)
    axes[1, 0].plot(full_hours[:48], input_sample, 'b-', 
                   label='Input (48h)', linewidth=2.5, alpha=0.8)
    axes[1, 0].plot(full_hours[48:], actual_real[sample_idx, :, 0], 'g-', 
                   label='Actual Future (24h)', linewidth=2.5, alpha=0.8)
    axes[1, 0].plot(full_hours[48:], pred_real[sample_idx, :, 0], 'r--', 
                   label='Predicted Future (24h)', linewidth=2.5, alpha=0.8)
    axes[1, 0].axvline(x=48, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    axes[1, 0].text(24, axes[1, 0].get_ylim()[1]*0.95, 'Past', 
                   ha='center', fontsize=10, style='italic')
    axes[1, 0].text(60, axes[1, 0].get_ylim()[1]*0.95, 'Future', 
                   ha='center', fontsize=10, style='italic')
    axes[1, 0].set_xlabel('Hour')
    axes[1, 0].set_ylabel('Temperature (°C)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Scatter plot
    axes[1, 1].set_title('Prediction Accuracy (All Timesteps)')
    sample_points = min(5000, pred_real.size)
    axes[1, 1].scatter(actual_real.flatten()[:sample_points], 
                      pred_real.flatten()[:sample_points], 
                      alpha=0.3, s=10, c='blue')
    min_temp = min(actual_real.min(), pred_real.min())
    max_temp = max(actual_real.max(), pred_real.max())
    axes[1, 1].plot([min_temp, max_temp], [min_temp, max_temp], 
                   'r--', linewidth=2, label='Perfect prediction')
    axes[1, 1].set_xlabel('Actual Temperature (°C)')
    axes[1, 1].set_ylabel('Predicted Temperature (°C)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].text(0.05, 0.95, f'MAE: {mae:.2f}°C', 
                   transform=axes[1, 1].transAxes, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('real_weather_recurrent_transformer.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved visualization to 'real_weather_recurrent_transformer.png'")
    
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Final Results:")
    print(f"   • Trained on real weather data from {len(cities)} cities")
    print(f"   • {len(X_train)} training sequences, {len(X_test)} test sequences")
    print(f"   • Mean Absolute Error: {mae:.2f}°C")
    print(f"   • Model can predict 24 hours ahead from 48 hours of history")
    print(f"\n🌡️  Temperature prediction accuracy: ±{mae:.2f}°C")


if __name__ == "__main__":
    main()