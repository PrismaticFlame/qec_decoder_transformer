# Block Recurrent Transformer

## Table of Contents
1. [The BRT](#the-brt)
2. [Why BRTs Were Created](#why-brts-were-created)
3. [How BRTs Work](#how-brts-work)
4. [Architecture Breakdown](#architecture-breakdown)
5. [Practical Applications](#practical-applications)
6. [Implementation Considerations](#implementation-considerations)
7. [Advantages and Limitations](#advantages-and-limitations)

## The BRT
From Google's *Nature* paper they use a "Recurrent Transformer based Neural Network", but what is this "Recurrent Transformer"?

Introduced in the paper *Block Recurrent Transformers* by Hutchins et al., this architecture applies a transformer layer in recurrent fashion along a sequence, and has linear complexity with respect to sequence length.

In simpler terms: BRTs combine the best of two worlds:
- **Transformers**: Powerful attention mechanisms that can look at all parts of the input simultaneously
- **Recurrent Neural Networks (RNNs)**: Ability to maintain memory and process sequences of arbitrary length

## Why BRTs Were Created

Traditional transformers have a major problem: **quadratic complexity**. When processing a sequence:
- A sequence of length N requires N² operations for self-attention
- Memory usage grows as N²
- This makes long sequences (thousands of tokens) computationally prohibitive

**Example**: Processing a sequence of 10,000 tokens requires 100,000,000 attention computations!

**Traditional RNNs** solve the length problem but have their own issues:
- Process one token at a time (slow, can't parallelize)
- Struggle with long-range dependencies
- Vanishing/exploding gradient problems

**BRTs solve both problems** by:
- Processing sequences in **blocks** (can parallelize within blocks like transformers)
- Maintaining **recurrent state** between blocks (handles long sequences like RNNs)
- Achieving **linear complexity** O(N) instead of quadratic O(N²)

## How BRTs Work

Think of BRTs like reading a book chapter by chapter, where you remember the key points as you go:

### The Basic Flow

1. **Split the input into blocks**
   - Instead of processing all 1000 tokens at once, split into 10 blocks of 100 tokens each
   - Each block is a manageable chunk

2. **Process each block with a transformer**
   - Within a block, use full transformer attention (all tokens attend to each other)
   - This is fast because each block is small

3. **Maintain state between blocks**
   - After processing block 1, create a "summary" (state vector)
   - Pass this state to block 2
   - Block 2 uses its tokens AND the state from block 1
   - Continue passing state forward through all blocks

### Visual Analogy

```
Book (Full Sequence)
├── Chapter 1 (Block 1) → Remember key points → State S₁
├── Chapter 2 (Block 2) → Uses S₁ + new info → State S₂
├── Chapter 3 (Block 3) → Uses S₂ + new info → State S₃
└── ...
```

You don't need to re-read all previous chapters to understand the current one; you just need the key points (state) from what came before.

## Architecture Breakdown

A BRT block has **two parallel processing paths**:

### 1. Token Path (Vertical/Transformer Direction)

This processes the **current block's tokens**:

```
Current Tokens (e.g., hours 0-12 of weather data)
    ↓
Self-Attention (tokens look at each other)
    ↓
Cross-Attention (tokens look at the state/memory)
    ↓
Feed-Forward Network
    ↓
Processed Tokens
```

**Purpose**: Understand the current information in context of what we've seen before

### 2. State Path (Horizontal/Recurrent Direction)

This updates the **memory/state** for the next block:

```
Previous State (summary of blocks 0 to N-1)
    ↓
Self-Attention (state attends to itself)
    ↓
Cross-Attention (state looks at current tokens)
    ↓
Gating (LSTM-like: decide what to remember/forget)
    ↓
Feed-Forward Network
    ↓
Updated State (summary of blocks 0 to N)
```

**Purpose**: Decide what information from the current block to remember for the future

### The Key Innovation: Bidirectional Cross-Attention

The magic happens in the cross-attention steps:
- **Tokens ← State**: "What do I need to know from the past to understand this?"
- **State ← Tokens**: "What from this block should I remember for later?"

This bidirectional information flow allows:
- Current processing to be informed by history
- Future processing to be informed by current observations

## Practical Applications

### 1. Weather Forecasting (Our Example)
- **Input**: 48 hours of temperature history
- **Blocks**: Split into 4 blocks of 12 hours each
- **State**: Carries patterns learned from earlier hours
- **Output**: Predict next 24 hours

**Why BRT works well**:
- Weather has temporal dependencies (today affects tomorrow)
- Can process long sequences (weeks/months of data)
- State captures seasonal patterns, trends

### 2. Quantum Error Correction (AlphaQubit)
- **Input**: Multiple rounds of syndrome measurements
- **Blocks**: Each round is a block
- **State**: Accumulates information about error patterns
- **Output**: Predict which qubits have errors

**Why BRT works well**:
- Errors are correlated over time
- Need to process hundreds of measurement rounds
- State learns which error patterns are likely

### 3. Other Applications
- **Language modeling**: Long documents (books, papers)
- **Time series**: Stock prices, sensor data, medical signals
- **Video processing**: Long videos split into segments
- **Audio**: Music generation, speech recognition

## Implementation Considerations

### Key Hyperparameters

```python
model = RecurrentTransformer(
    input_dim=1,        # Number of features per timestep
    hidden_dim=64,      # Size of token representations
    state_dim=32,       # Size of recurrent state (memory)
    num_blocks=4,       # How many blocks in the architecture
    block_size=12,      # Tokens per block
    num_heads=4,        # Multi-head attention heads
    dropout=0.1         # Regularization
)
```

**Rules of thumb**:
- `block_size`: 10-50 tokens (small enough for fast attention)
- `hidden_dim`: 64-512 (larger = more capacity, slower training)
- `state_dim`: Usually 0.25-0.5 × hidden_dim
- `num_blocks`: Sequence_length / block_size

### Training Tips

1. **Start small**: Test with tiny models first
2. **Use gradient clipping**: Recurrent models can have exploding gradients
3. **Learning rate**: Start with 0.001, reduce if loss explodes
4. **Batch size**: Larger batches (32-128) work better
5. **Epochs**: 50-200 depending on data size

### Memory Considerations

**Memory usage** ≈ batch_size × block_size² × hidden_dim

For our weather example:
- batch_size=32, block_size=12, hidden_dim=64
- Memory ≈ 32 × 144 × 64 = ~300KB per forward pass
- Much better than full transformer: 32 × 48² × 64 = ~4.7MB

## Advantages and Limitations

### Advantages ✅

1. **Linear complexity**: Can process very long sequences
   - Traditional transformer: O(N²)
   - BRT: O(N)
   
2. **Parallelization within blocks**: Fast training
   - Process all tokens in a block simultaneously
   - Unlike RNNs which are strictly sequential

3. **Flexible context**: State size is fixed
   - Can process sequences of any length
   - Memory usage doesn't grow with sequence length

4. **Strong performance**: Competitive with full transformers
   - Maintains ability to capture long-range dependencies
   - State vector acts as compressed memory

5. **Efficient inference**: Less computation at test time
   - Don't need to process entire history
   - Just maintain and update state

### Limitations ⚠️

1. **Not as powerful as full transformers on short sequences**
   - If your sequences are short (<500 tokens), regular transformers may be better
   - Trade some accuracy for efficiency

2. **State is a bottleneck**
   - Information must flow through fixed-size state
   - Some information may be lost if state is too small

3. **Less parallelization than full transformers**
   - Must process blocks sequentially
   - Can't parallelize across blocks (only within)

4. **Tuning complexity**
   - More hyperparameters to tune (block_size, state_dim)
   - Need to balance block size vs. number of blocks

5. **Implementation complexity**
   - More complex than standard transformers
   - Requires careful handling of state passing

### When to Use BRTs

**Use BRTs when**:
- ✅ Sequences are long (>1000 tokens)
- ✅ Real-time processing needed
- ✅ Limited memory/compute
- ✅ Sequential dependencies are important

**Use standard transformers when**:
- ❌ Sequences are short (<500 tokens)
- ❌ Maximum accuracy is critical
- ❌ Compute resources are unlimited
- ❌ Can precompute entire sequence

## Comparison Summary

| Feature | Transformer | RNN | BRT |
|---------|------------|-----|-----|
| **Complexity** | O(N²) | O(N) | O(N) |
| **Parallelization** | Full | None | Partial (within blocks) |
| **Long sequences** | ❌ Expensive | ✅ Good | ✅ Good |
| **Long-range deps** | ✅ Excellent | ⚠️ Difficult | ✅ Good |
| **Memory usage** | High | Low | Medium |
| **Training speed** | Fast | Slow | Medium-Fast |
| **Implementation** | Simple | Simple | Complex |

## Further Reading

- **Original Paper**: "Block-Recurrent Transformers" (Hutchins et al., 2022)
- **AlphaQubit**: "Quantum Error Correction with Neural Networks" (Google DeepMind, 2023)
- **Related Work**: 
  - Transformer-XL (segment-level recurrence)
  - Compressive Transformers (compressed memory)
  - Linear Transformers (efficient attention variants)

## Getting Started

To implement your own BRT:

1. **Start with the NumPy version** (educational)
   - Understand each component
   - See how state flows through blocks
   - Visualize attention patterns

2. **Move to PyTorch** (production)
   - Use automatic differentiation
   - Leverage GPU acceleration
   - Train on real datasets

3. **Experiment with hyperparameters**
   - Try different block sizes
   - Adjust state dimensions
   - Monitor training curves

4. **Apply to your domain**
   - Adapt input/output dimensions
   - Customize for your data type
   - Add domain-specific features

---

**Key Takeaway**: Block Recurrent Transformers give you the power of transformers' attention mechanisms with the efficiency of RNNs' linear complexity. They're the sweet spot for processing long sequences where full transformers are too expensive but you still need strong performance.