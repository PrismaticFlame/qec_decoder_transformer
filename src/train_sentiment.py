"""
IMDB Sentiment Analysis with Transformer
Trains a transformer model to classify movie reviews as positive or negative
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import os

# Set device that will do the mathing
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # If you have an NVIDIA GPU, your GPU will get used. Otherwise, CPU
print(f"Using device: {device}")

# Hyperparameters:
MAX_LENGTH = 256  # Max tokens per review
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.1

# Create directories for where the model will be saved and where the data will be saved
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

print("Loading IMDB dataset...")
# Load IMDB dataset, use 25k to train and 25k to test
dataset = load_dataset('imdb')
train_data = dataset['train']
test_data = dataset['test']

print(f"Train samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")

# Load tokenizer (using DistilBERT tokenizer for vocabulary)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

class IMDBDataset(Dataset):
    """Custom Dataset for IMDB reviews"""
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]['text']
        label = self.data[idx]['label']

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Create datasets and dataloaders from above class
print("Creating datasets...")
train_dataset = IMDBDataset(train_data, tokenizer, MAX_LENGTH)
test_dataset = IMDBDataset(test_data, tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class SentimentTransformer(nn.Module):
    """Transformer model for sentiment classification"""
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_length, num_classes=2, dropout=0.1):
        super(SentimentTransformer, self).__init__()  # Some super class that I did not define that came preloaded with PyTorch. Will need to read documentation.and

        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.shape

        # Create embeddings
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(positions)
        x = token_embeds + pos_embeds

        # Create attention mask for transformer (True = masked position)
        mask = (attention_mask == 0)

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)

        # Use [CLS] token (first token) for classification
        x = x[:, 0, :]
        x = self.dropout(x)
        logits = self.fc(x)

        return logits


# Initialize model
print("Initializing model...")
vocab_size = tokenizer.vocab_size
model = SentimentTransformer(
    vocab_size=vocab_size,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    max_length=MAX_LENGTH,
    dropout=DROPOUT
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})

    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), accuracy, all_preds, all_labels

# Training loop
print("\nStarting training...")
train_losses = []
train_accs = []
test_losses = []
test_accs = []

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Evaluate
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

# Final eval
print("\n" + "="*50)
print("Final Result")
print("\n" + "="*50)
print(classification_report(test_labels, test_preds, target_names=['Negative', 'Positive']))

# Save model
model_path = 'models/sentiment_transformer.pt'
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab_size': vocab_size,
    'embed_dim': EMBED_DIM,
    'num_heads': NUM_HEADS,
    'num_layers': NUM_LAYERS,
    'max_length': MAX_LENGTH,
}, model_path)
print(f"\nModel saved to {model_path}")

# Plot training curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(test_accs, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('data/training_curves.png', dpi=150, bbox_inches='tight')
print("Training curves saved to data/training_curves.png")

# Test with custom reviews
def predict_sentiment(text, model, tokenizer, device, max_length=256):
    """Predict sentiment of a single text"""
    model.eval()
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()

    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = probs[0][prediction].item()

    return sentiment, confidence

# Test examples
print("\n" + "="*50)
print("TESTING WITH CUSTOM REVIEWS")
print("="*50)

test_reviews = [
    "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout.",
    "Terrible waste of time. The storyline made no sense and the acting was wooden.",
    "It was okay, nothing special but not terrible either.",
    "One of the best films I've ever seen! Highly recommend!",
    "I fell asleep halfway through. Boring and predictable."
]

for review in test_reviews:
    sentiment, confidence = predict_sentiment(review, model, tokenizer, device)
    print(f"\nReview: {review[:80]}...")
    print(f"Prediction: {sentiment} (confidence: {confidence:.2%})")

print("\n" + "="*50)
print("Training complete! You can now use the model to classify movie reviews.")
print("="*50)


