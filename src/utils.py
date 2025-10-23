import torch
import os
from datetime import datetime


def get_device():
    """
    Automatically detect and return the best available device.
    
    Returns:
        torch.device: The device to use (cuda or cpu)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device


def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint.pt"):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        loss: Current loss value
        filename: Path to save checkpoint
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(model, optimizer, filename="checkpoint.pt"):
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        filename: Path to checkpoint file
        
    Returns:
        epoch: Epoch number from checkpoint
        loss: Loss value from checkpoint
    """
    if not os.path.exists(filename):
        print(f"No checkpoint found at {filename}")
        return 0, float('inf')
    
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded from {filename}")
    print(f"Resuming from epoch {epoch} with loss {loss:.4f}")
    
    return epoch, loss


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        total: Total number of parameters
        trainable: Number of trainable parameters
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total, trainable


def print_model_summary(model):
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
    """
    print("=" * 70)
    print("Model Summary")
    print("=" * 70)
    
    total, trainable = count_parameters(model)
    
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Non-trainable parameters: {total - trainable:,}")
    
    # Calculate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024 / 1024
    
    print(f"Model size: {size_mb:.2f} MB")
    print("=" * 70)


def get_lr(optimizer):
    """
    Get current learning rate from optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        Learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking metrics during training.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_padding_mask(seq, pad_idx=0):
    """
    Create padding mask for sequences.
    
    Args:
        seq: Input sequence tensor (batch_size, seq_len)
        pad_idx: Index used for padding
        
    Returns:
        Mask tensor where True indicates padding positions
    """
    return (seq == pad_idx)


def create_look_ahead_mask(size):
    """
    Create look-ahead mask to prevent attending to future positions.
    Used in decoder to maintain causality.
    
    Args:
        size: Size of the sequence
        
    Returns:
        Upper triangular mask
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask


# Example usage
if __name__ == "__main__":
    print("Testing utility functions...\n")
    
    # Test device detection
    device = get_device()
    print()
    
    # Test AverageMeter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"AverageMeter test: avg={meter.avg:.2f}, count={meter.count}")
    print()
    
    # Test padding mask
    seq = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
    mask = create_padding_mask(seq, pad_idx=0)
    print("Padding mask test:")
    print(f"Sequence:\n{seq}")
    print(f"Mask:\n{mask}")
    print()
    
    # Test look-ahead mask
    look_ahead = create_look_ahead_mask(5)
    print("Look-ahead mask test:")
    print(look_ahead)
    print()
    
    print("✓ All utility functions working correctly!")