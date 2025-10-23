import torch
import torch.nn as nn
import math

class SimpleTransformer(nn.Module):
    """
    A simple Transformer model for sequence-to-sequence tasks.
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Dimension of model embeddings (default: 512)
        nhead: Number of attention heads (default: 8)
        num_layers: Number of transformer layers (default: 6)
        dim_feedforward: Dimension of feedforward network (default: 2048)
        dropout: Dropout rate (default: 0.1)
        max_seq_len: Maximum sequence length (default: 5000)
    """
    
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, 
                 dim_feedforward=2048, dropout=0.1, max_seq_len=5000):
        super(SimpleTransformer, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layer
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, src_mask=None):
        """
        Forward pass
        
        Args:
            src: Input tensor of shape (batch_size, seq_len)
            src_mask: Optional mask for padding
            
        Returns:
            Output tensor of shape (batch_size, seq_len, vocab_size)
        """
        # Embedding and positional encoding
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # Pass through transformer
        output = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        
        # Project to vocabulary
        output = self.fc_out(output)
        
        return output
    
    def generate(self, start_tokens, max_length=50, temperature=1.0):
        """
        Generate sequence autoregressively
        
        Args:
            start_tokens: Starting tokens (batch_size, start_len)
            max_length: Maximum length to generate
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Generated sequence
        """
        self.eval()
        with torch.no_grad():
            current_seq = start_tokens
            
            for _ in range(max_length - start_tokens.size(1)):
                # Get predictions for current sequence
                output = self.forward(current_seq)
                
                # Get next token probabilities (last position)
                next_token_logits = output[:, -1, :] / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                current_seq = torch.cat([current_seq, next_token], dim=1)
            
            return current_seq


class PositionalEncoding(nn.Module):
    """
    Positional encoding to inject position information into embeddings.
    Uses sine and cosine functions of different frequencies.
    """
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# Example usage and testing
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Hyperparameters
    vocab_size = 10000
    d_model = 256
    nhead = 8
    num_layers = 4
    batch_size = 2
    seq_len = 20
    
    # Create model
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=1024,
        dropout=0.1
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=" * 50)
    print("Simple Transformer Model")
    print("=" * 50)
    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Model dimension: {d_model}")
    print(f"Number of heads: {nhead}")
    print(f"Number of layers: {num_layers}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 50)
    
    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Input shape: {dummy_input.shape}")
    
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: (batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size})")
    
    # Test generation
    print("\nTesting generation...")
    start_tokens = torch.randint(0, vocab_size, (1, 5))
    generated = model.generate(start_tokens, max_length=15)
    print(f"Start tokens shape: {start_tokens.shape}")
    print(f"Generated sequence shape: {generated.shape}")
    print(f"Generated tokens: {generated[0].tolist()}")
    
    print("\n✓ All tests passed!")