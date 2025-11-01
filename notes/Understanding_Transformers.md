# Understanding Transformers: Architecture & Implementation Guide

## Table of Contents
1. [What is a Transformer?](#what-is-a-transformer)
2. [Core Components](#core-components)
3. [Code Implementation Breakdown](#code-implementation-breakdown)
4. [How It All Works Together](#how-it-all-works-together)

---

## What is a Transformer?

A **Transformer** is a neural network architecture introduced in the 2017 paper "Attention is All You Need" by Vaswani et al. Unlike previous models (RNNs, LSTMs) that process sequences step-by-step, transformers process entire sequences at once using **attention mechanisms**.

### Key Innovation: Self-Attention
The transformer can look at all words in a sentence simultaneously and figure out which words are most relevant to understanding each other word. For example, in "The animal didn't cross the street because it was too tired", the model learns that "it" refers to "animal", not "street".

---

## Core Components

### 1. Input Embeddings

**What it does:** Converts words/tokens into dense vectors (arrays of numbers) that capture semantic meaning.

**In our code:**
```python
self.token_embedding = nn.Embedding(vocab_size, embed_dim)
```

**Example:**
- Word "good" → `[0.2, -0.5, 0.8, ..., 0.1]` (128 numbers)
- Word "bad" → `[-0.3, 0.4, -0.7, ..., -0.2]` (128 numbers)

Similar words get similar vectors!

---

### 2. Positional Encoding

**What it does:** Adds information about word position in the sentence. Without this, "dog bites man" and "man bites dog" would look the same to the model!

**In our code:**
```python
self.position_embedding = nn.Embedding(max_length, embed_dim)
positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
pos_embeds = self.position_embedding(positions)
x = token_embeds + pos_embeds  # Combine token and position info
```

**How it works:**
- Position 0 (first word) → `[0.1, 0.3, -0.2, ...]`
- Position 1 (second word) → `[0.2, -0.1, 0.4, ...]`
- These are added to the token embeddings

---

### 3. Multi-Head Attention (The Magic!)

**What it does:** The core of the transformer. It allows each word to "attend to" (look at) other words in the sentence to understand context.

**In our code:**
```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=embed_dim,        # Size of embeddings (128)
    nhead=num_heads,          # Number of attention heads (4)
    dim_feedforward=embed_dim * 4,  # Size of feedforward network
    dropout=dropout,
    batch_first=True
)
```

**How Multi-Head Attention Works:**

Imagine you're reading: "The cat sat on the mat because it was soft"

- **Head 1** might learn: "it" → "mat" (grammatical relationships)
- **Head 2** might learn: "cat" → "sat" (subject-verb relationships)
- **Head 3** might learn: "soft" → "mat" (descriptive relationships)
- **Head 4** might learn: "because" → "soft" (causal relationships)

Each head learns different types of relationships!

**The Attention Mechanism (Simplified):**

For each word, attention computes:
1. **Query (Q):** "What am I looking for?"
2. **Key (K):** "What do I contain?"
3. **Value (V):** "What information do I provide?"

```
Attention(Q, K, V) = softmax(Q * K^T / √d) * V
```

This formula calculates how much each word should pay attention to every other word.

---

### 4. Feed-Forward Network

**What it does:** After attention, each word's representation goes through a small neural network to transform the information.

**In our code:**
```python
dim_feedforward=embed_dim * 4  # 128 * 4 = 512
```

**Structure:**
```
Input (128) → Linear(512) → ReLU → Linear(128) → Output (128)
```

This helps the model learn complex non-linear patterns.

---

### 5. Layer Normalization & Residual Connections

**What it does:** 
- **Layer Norm:** Stabilizes training by normalizing values
- **Residual Connections:** Allows information to flow directly through layers (helps with deep networks)

**In our code:**
These are built into `nn.TransformerEncoderLayer` automatically!

**Structure:**
```
x_new = LayerNorm(x + Attention(x))       # Add & Norm
x_final = LayerNorm(x_new + FeedForward(x_new))  # Add & Norm
```

The `+` signs are the residual connections - they prevent information loss in deep networks.

---

### 6. Transformer Encoder Stack

**What it does:** Stacks multiple encoder layers on top of each other. Each layer refines the understanding.

**In our code:**
```python
self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
```

With `num_layers=2`, we have:
```
Input → [Encoder Layer 1] → [Encoder Layer 2] → Output
```

Each layer contains:
- Multi-head attention
- Feed-forward network
- Layer normalization
- Residual connections

---

### 7. Classification Head

**What it does:** Takes the final encoded representation and predicts the output (positive/negative sentiment).

**In our code:**
```python
# Use first token ([CLS]) for classification
x = x[:, 0, :]  # Take first token's embedding
x = self.dropout(x)
logits = self.fc(x)  # Linear layer: 128 → 2 (negative, positive)
```

The first token acts as a summary of the entire sentence.

---

## Code Implementation Breakdown

### Full Forward Pass

Let's trace what happens to the sentence "This movie is great!":

```python
def forward(self, input_ids, attention_mask):
    batch_size, seq_len = input_ids.shape
    # input_ids shape: [batch_size, 256] - tokenized sentence
    
    # Step 1: Token Embeddings
    token_embeds = self.token_embedding(input_ids)
    # Shape: [batch_size, 256, 128]
    # Each of 256 tokens → 128-dimensional vector
    
    # Step 2: Position Embeddings
    positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
    pos_embeds = self.position_embedding(positions)
    # Shape: [batch_size, 256, 128]
    
    # Step 3: Combine Embeddings
    x = token_embeds + pos_embeds
    # Shape: [batch_size, 256, 128]
    
    # Step 4: Create Attention Mask
    # True = ignore this position (padding)
    mask = (attention_mask == 0)
    
    # Step 5: Transformer Encoding
    x = self.transformer(x, src_key_padding_mask=mask)
    # Shape: [batch_size, 256, 128]
    # Each token now has context from all other tokens!
    
    # Step 6: Extract [CLS] token (first token)
    x = x[:, 0, :]
    # Shape: [batch_size, 128]
    # This summarizes the entire sentence
    
    # Step 7: Classification
    x = self.dropout(x)
    logits = self.fc(x)
    # Shape: [batch_size, 2]
    # [score_negative, score_positive]
    
    return logits
```

---

## How It All Works Together

### Example: Sentiment Analysis Process

**Input:** "This movie was absolutely fantastic!"

**Step-by-step:**

1. **Tokenization** (before model):
   ```
   ["[CLS]", "this", "movie", "was", "absolutely", "fantastic", "!", "[SEP]", "[PAD]", ...]
   → [101, 2023, 3185, 2001, 7078, 13797, 999, 102, 0, 0, ...]
   ```

2. **Embedding** (in model):
   ```
   Each token ID → 128-dimensional vector
   Position 0 → add position embedding 0
   Position 1 → add position embedding 1
   ...
   ```

3. **Attention Layer 1**:
   ```
   "fantastic" pays attention to: "movie" (70%), "absolutely" (20%), others (10%)
   "was" pays attention to: "movie" (50%), "fantastic" (40%), others (10%)
   Each word now has context!
   ```

4. **Feed-Forward**:
   ```
   Each word's representation goes through neural network
   Transforms: 128 → 512 → 128
   Non-linear patterns learned
   ```

5. **Attention Layer 2**:
   ```
   Even more refined understanding
   "fantastic" now understands it's describing "movie" in a positive way
   ```

6. **Classification**:
   ```
   [CLS] token has absorbed information from entire sentence
   [CLS] → Dropout → Linear → [0.1, 4.8]
                                 ↓
   Softmax → [0.01, 0.99] = 99% Positive!
   ```

---

## Key Hyperparameters in Our Model

| Parameter | Value | What it means |
|-----------|-------|---------------|
| `EMBED_DIM` | 128 | Size of word vectors |
| `NUM_HEADS` | 4 | Number of attention mechanisms |
| `NUM_LAYERS` | 2 | Number of encoder layers stacked |
| `MAX_LENGTH` | 256 | Maximum number of tokens per review |
| `DROPOUT` | 0.1 | 10% of neurons randomly turned off (prevents overfitting) |

---

## Attention Visualization (Conceptual)

For sentence: "The cat sat on the mat"

```
        The    cat    sat    on     the    mat
The    [0.2]  [0.1]  [0.05] [0.05] [0.5]  [0.1]
cat    [0.1]  [0.4]  [0.3]  [0.05] [0.05] [0.1]
sat    [0.05] [0.4]  [0.3]  [0.15] [0.05] [0.05]
on     [0.1]  [0.05] [0.2]  [0.2]  [0.1]  [0.35]
the    [0.3]  [0.05] [0.05] [0.1]  [0.2]  [0.3]
mat    [0.1]  [0.1]  [0.05] [0.3]  [0.2]  [0.25]
```

Numbers show attention weights (how much each word focuses on others).
- "cat" pays most attention to itself and "sat" (subject-verb)
- "on" pays most attention to "mat" (preposition-object)

---

## Why Transformers Are Powerful

1. **Parallel Processing:** All words processed simultaneously (unlike RNNs which go one-by-one)
2. **Long-Range Dependencies:** Can connect words far apart in a sentence
3. **Interpretable:** Attention weights show what the model is "looking at"
4. **Scalable:** Works for small tasks and massive tasks (like GPT-4)

---

## Training Process

```
For each batch of reviews:
    1. Forward pass: Predict sentiment
    2. Calculate loss: How wrong were we?
    3. Backward pass: Calculate gradients
    4. Update weights: Make model slightly better
    
Repeat for 3 epochs (3 full passes through all data)
```

**Loss Function:** Cross-Entropy Loss
- Measures how different predictions are from true labels
- Model tries to minimize this loss

**Optimizer:** AdamW
- Smart way to update model weights
- Adjusts learning rate automatically for each parameter

---

## Differences from Original Transformer

Our implementation is **encoder-only** (for classification). The original transformer has:

- **Encoder:** Understands input (what we have)
- **Decoder:** Generates output (used for translation, text generation)

For sentiment analysis, we only need the encoder!

---

## Further Reading

- **Original Paper:** "Attention is All You Need" (Vaswani et al., 2017)
- **BERT:** Encoder-only transformer (what our model is similar to)
- **GPT:** Decoder-only transformer (for text generation)
- **T5, BART:** Encoder-decoder transformers (for translation, summarization)

---

## Experimenting with the Code

Try changing these parameters in the code:

```python
EMBED_DIM = 256      # Bigger embeddings (more expressive)
NUM_HEADS = 8        # More attention heads (more patterns)
NUM_LAYERS = 4       # Deeper network (more understanding)
MAX_LENGTH = 512     # Longer sequences (more context)
```

**Trade-off:** Bigger models are more powerful but slower to train and need more memory!

---

## Summary

A Transformer is like a smart reader that:
1. **Reads** all words at once (embeddings)
2. **Understands** relationships between words (attention)
3. **Refines** understanding through multiple layers
4. **Decides** on a final answer (classification head)

The key innovation is **attention** - allowing every word to look at every other word and decide what's important!

