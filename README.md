# 🦀 Oxidized-GPT: Zero-Dependency Rust Implementation

A compact, educational GPT implementation written in Rust using only the standard library. This project is a Rust port of [microGPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) by Andrej Karpathy, demonstrating a full stack: custom RNG, a minimal autodiff engine, a transformer forward+backward pass, Adam optimizer, training loop, and CPU inference — all without external crates.

## Credits

This implementation is based on Andrej Karpathy's [microGPT blog post](https://karpathy.github.io/2026/02/12/microgpt/), which presents a minimal, educational GPT implementation in Python. Oxidized-GPT faithfully translates the core concepts to Rust while maintaining zero external dependencies (standard library only).

This workspace uses the canonical Rust layout (`Cargo.toml` + `src/main.rs`).

## Key points (quick)

- Zero external dependencies (std only)
- Autodiff graph with basic ops (Add, Mul, Pow, Exp, Log, ReLU)
- Higher-level operations built from primitives (matmul, softmax, cross-entropy)
- Transformer with RMSNorm, multi-head attention, and residuals
- Parameter tracking to extract gradients and ADAM optimizer
- CPU-only inference with autoregressive generation

## Architecture

### Overview

Oxidized-GPT implements a character-level language model using the GPT (Generative Pre-trained Transformer) architecture. The model is trained on a dataset of names and learns to generate new, plausible names character by character.

### Core Components

#### 1. **Custom Automatic Differentiation Engine**

The implementation includes a full computational graph system (`Graph`) that tracks operations during the forward pass and computes gradients in the backward pass:

- **Basic Graph Operations**: Add, Multiply, Power, Exp, Log, ReLU
- **Higher-Level Operations**: MatMul, Softmax, and cross-entropy loss built from basic primitives
- **Gradient Computation**: Topological sorting ensures gradients propagate correctly through the graph
- **Memory Management**: Graph nodes track dependencies; graphs are periodically reset to bound memory

#### 2. **Transformer Architecture**

The model follows the standard decoder-only transformer architecture with:

- **Token + Position Embeddings**: Learnable embeddings for each character and position
- **Transformer Blocks** (configurable depth):
  - **RMSNorm**: Root Mean Square Layer Normalization for stable training
  - **Multi-Head Self-Attention**: Parallel attention heads with causal masking
  - **MLP Block**: Two-layer feed-forward network with ReLU activation
  - **Residual Connections**: Skip connections around attention and MLP blocks
- **Output Head**: Linear projection to vocabulary logits + softmax

#### 3. **Attention Mechanism**

- **Causal Self-Attention**: Each token only attends to previous tokens (autoregressive property)
- **Multi-Head**: Multiple attention heads learn different representation subspaces
- **KV Cache**: During inference, past keys and values are cached to avoid recomputation

#### 4. **Training System**

- **Adam Optimizer**: First and second moment estimates with bias correction
- **Learning Rate Schedule**: Linear decay from initial rate to zero
- **Mini-batch Training**: Processes batches of sequences in parallel
- **Cross-Entropy Loss**: Standard language modeling objective

#### 5. **Custom RNG**

- **Linear Congruential Generator (LCG)**: Fast, deterministic pseudo-random number generation
- **Box-Muller Transform**: Converts uniform random samples to Gaussian distribution
- **Seeded Generation**: Reproducible results for debugging and testing

### Model Parameters

For a typical configuration:
- `n_embd = 16`: Embedding dimension
- `n_head = 4`: Number of attention heads
- `n_layer = 1`: Number of transformer layers
- `block_size = 16`: Maximum sequence length
- `vocab_size = 27`: Character vocabulary size (26 letters + BOS token)

Total parameters: ~51K (varies with configuration)

### Inference

The model generates text autoregressively:
1. Start with a BOS (beginning of sentence) token
2. Forward pass produces logits for next character
3. Temperature scaling and categorical sampling select next token
4. Append token and repeat until EOS or max length
5. KV cache optimization speeds up generation

## Files of interest

- `Cargo.toml` — project manifest
- `src/main.rs` — complete implementation (main training + inference)

## Quick start

1. Download the dataset (names list used by the code):

```bash
curl -o input.txt https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt
```

2. Build & run (use release for performance):

```bash
cargo run --release
```

If `input.txt` is not present the program will print a helpful message and exit.

## Configuration (in code)

Open `src/main.rs` and inspect the `Config` used by `main`:

```rust
let config = Config {
    vocab_size,      // auto-detected from the dataset (+1 for BOS)
    n_embd: 16,      // embedding dimension
    n_head: 4,       // attention heads
    n_layer: 1,      // transformer layers
    block_size: 16,  // max sequence length
};
```

Training in the shipped example uses:

- `num_steps = 1000`
- initial `learning_rate = 0.005` (linearly decays)
- `beta1 = 0.9`, `beta2 = 0.999`, `eps = 1e-8` (Adam default-like)

These are set inside `train()` in `src/main.rs` and can be changed directly.

## What to expect when running

Typical console output prints dataset and model sizes, then training progress (loss) and, after training, inference samples. Example lines:

```
🦀 Oxidized-GPT: Zero-Dependency Pure Rust Implementation

num docs: 32033
vocab size: 27
num params: 4192

Training...
step    1 / 1000 | loss 3.3057
step  100 / 1000 | loss 1.9333
step  200 / 1000 | loss 2.3840
step  300 / 1000 | loss 3.1924
step  400 / 1000 | loss 2.0515
step  500 / 1000 | loss 2.1795
step  600 / 1000 | loss 2.8178
step  700 / 1000 | loss 2.8253
step  800 / 1000 | loss 2.6771
step  900 / 1000 | loss 2.9577
step 1000 / 1000 | loss 2.7220

--- inference (new, hallucinated names) ---
sample  1: anmie
sample  2: lyase
sample  3: iavina
sample  4: evty
sample  5: khiri
sample  6: smaenalaz
sample  7: avezin
sample  8: conee
sample  9: gielalia
sample 10: yeahi
sample 11: melio
sample 12: jakyti
sample 13: byani
sample 14: esma
sample 15: naldi
sample 16: emidi
sample 17: ahyanaut
sample 18: keida
sample 19: dales
sample 20: lrymi
```

## Implementation notes (high level)

- **RNG**: Custom LCG + Box-Muller for normal samples; deterministic given seed.
- **Autodiff**: `Graph` builds nodes during forward pass; `graph.backward(root)` computes gradients via topological ordering.
- **Parameters**: `Parameters` stores `data`, `grads`, optimizer state (`m`, `v`), and tracks `leaf_indices` so gradients can be aggregated for each learned parameter.
- **Attention**: Keys/values are cached per layer (`KVCache`) and inference uses cached past K/V vectors for autoregressive generation.
- **Inference**: `generate()` builds a graph per step, uses temperature scaling and categorical sampling, and resets the graph every 4 steps to bound memory.
- **Gradient Clipping**: Gradients are clipped during backpropagation (±10 for node gradients, ±5 for parameter gradients) to prevent instability.

## Development and experiments

- To increase model capacity, raise `n_embd`, `n_head`, and `n_layer` in `src/main.rs`.
- For faster local builds with CPU optimizations:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

- To improve sample quality, increase training steps or model size, or experiment with temperature/top-k sampling.

## Notes & caveats

- This implementation is educational and CPU-only — there is no GPU acceleration.
- The custom RNG is fast and deterministic but not cryptographically secure.
- Some operations (e.g., matrix routines) are implemented in straightforward Rust for clarity, not maximum throughput.

## Contributing

Contributions are welcome! Suggested improvements:

- SIMD/vectorized matmul
- Model save/load (serialization)
- More sampling strategies (top-k, top-p)
- Training/benchmark harness
- GPU acceleration via compute shaders or CUDA bindings

Please open an issue or pull request on GitHub.

## References

- **Original microGPT Blog Post**: [https://karpathy.github.io/2026/02/12/microgpt/](https://karpathy.github.io/2026/02/12/microgpt/) by Andrej Karpathy

## License

MIT — see `LICENSE` (or assume MIT for the example code in this repository).


