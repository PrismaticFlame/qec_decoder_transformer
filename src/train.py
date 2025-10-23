import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

from transformer import SimpleTransformer
from utils import get_device, save_checkpoint, load_checkpoint


class DummySequenceDataset(Dataset):
    """
    A simple dummy dataset for testing.
    Generates random sequences where the target is input shifted by 1.
    """

    def __init__(self, num_samples=1000, seq_len=50, vocab_size=10000):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random sequence
        sequence = torch.randint(0, self.vocab_size, (self.seq_len + 1,))

        # Input is all but last token, target is all but first token
        src = sequence[:-1]
        tgt = sequence[1:]

        return src, tgt
    

def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    Train for one epoch
    """
    model.train()
    total_loss = 0

    with tqdm(train_loader, desc="Training", unit="batch") as pbar:
        for batch_idx, (src, tgt) in enumerate(pbar):
            src, tgt = src.to(device), tgt.to(device)

            # Forward pass
            optimizer.zero_grad()
            output = model(src)

            # Reshape for loss calculation
            # output: (batch, seq_len, vocab_size) -> (batch * seq_len, vocab_size)
            # tgt: (batch, seq_len) -> (batch * seq_len)
            output = output.reshape(-1, output.size(-1))
            tgt = tgt.reshape(-1)

            # Calculate loss
            loss = criterion(output, tgt)

            # Backward pass
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Update metrics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)

            # Update progress bar
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
        
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    """
    Validate the model
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        with tqdm(val_loader, desc="Validation", unit="batch") as pbar:
            for src, tgt in pbar:
                src, tgt = src.to(device), tgt.to(device)

                # Forward pass
                output = model(src)

                # Reshape for loss calc
                output = output.reshape(-1, output.size(-1))
                tgt = tgt.reshape(-1)

                # Calc loss
                loss = criterion(output, tgt)
                total_loss += loss.item()

                # Update progress bar
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(val_loader)

def main():
    # Hyperparameters
    vocab_size = 10000
    d_model = 256
    nhead = 8
    num_layers = 4
    dim_feedforward = 1024
    dropout = 0.1
    
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.0001
    
    seq_len = 50
    num_train_samples = 1000
    num_val_samples = 200

    # Device
    device = get_device()
    print(f"\nUsing device: {device}")

    # Create datasets
    print("Creating datasets...")
    train_dataset = DummySequenceDataset(num_train_samples, seq_len, vocab_size)
    val_dataset = DummySequenceDataset(num_val_samples, seq_len, vocab_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    print("Creating model...")
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    ).to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_val_loss = float('inf')

    print("Starting training...\n")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}\n")

        # Save checkpoint if best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                filename="models/best_model.pt"
            )
            print(f"Saved best model (val_loss: {val_loss:.4f})\n")
    
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Test generation
    print("\nTesting text generation...")
    model.eval()
    start_tokens = torch.randint(0, vocab_size, (1, 5)).to(device)
    generated = model.generate(start_tokens, max_length=20)
    print(f"Generated sequence: {generated[0].tolist()}")


if __name__ == "__main__":
    main()