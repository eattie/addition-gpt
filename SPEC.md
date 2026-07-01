# Project: A GPT That Learns Addition (From Scratch)

Build a tiny transformer that learns to add. No video, no following along. You look
things up only when stuck on one specific thing. The output is verifiable: it either
learns to add or it doesn't.

Stack: Python + PyTorch. (Feinberg's version uses JAX; use PyTorch here since it's more
forgiving while you're solidifying fundamentals. Port to JAX later.)

**Rule for the whole project:** every checkpoint has a `Predict:` line with a blank. Before
you run anything, fill in the shape you expect — actually type it in. Then run, print the
real shape, and compare. When you're wrong, stop and find out why before continuing. The gap
between your guess and reality is the actual point of the project; don't skip the guess to
peek at the answer.

---

## The task

Teach the model to complete addition problems presented as strings:

```
"12+37=49"
```

The model sees characters left to right and predicts the next character. If it learns the
pattern, feeding it `"12+37="` makes it produce `4`, then `9`.

Vocabulary is 13 characters: digits `0-9`, plus `+`, `=`, and a padding token.

---

## Build order

Build in this order. **Do not move to the next piece until the current one passes its
checkpoint.** Each checkpoint is a thing you can run and see.

### 1. The data

Generate examples of the form `a+b=c` where `a` and `b` are 1–2 digit numbers.

- Pick a fixed max length and pad every string to it (padding makes batching trivial).
- Build a char→int mapping and its inverse.
- Encode strings to integer tensors.

**Checkpoint:** print 5 random encoded examples and decode them back. They should round-trip
exactly.
- `Predict:` shape of one encoded example = `__________`
- `Predict:` shape of a batch of them = `__________`

### 2. The batching / targets

For next-char prediction, the target is the input shifted by one position. Input is
characters `0..n-1`, target is characters `1..n`.

- Write a function that returns `(x, y)` batches where `y` is `x` shifted left by one.

**Checkpoint:** print one `(x, y)` pair, decoded. Confirm by eye that `y` is `x` shifted by
one.
- `Predict:` shape of `x` = `__________`
- `Predict:` shape of `y` = `__________`

### 3. Token + position embeddings

- Token embedding: a lookup table mapping each of the 13 token IDs to a vector of size
  `d_model`.
- Position embedding: a table mapping each position `0..seq_len-1` to a vector of size
  `d_model`.
- The input to the rest of the model is the sum of the two.

**Checkpoint:** feed in one batch, print the shape after embedding.
- `Predict:` shape after embedding = `__________`  (before running — don't peek at later steps)

### 4. A single attention head

This is the core. Build it slowly. For one head:

- Three linear projections of the embedded input: Q, K, V.
- Attention scores: `Q @ K.transpose` — every position scored against every other.
- Scale the scores by `1/sqrt(head_dim)`.
- **Causal mask:** position `i` may only attend to positions `≤ i`. Set the upper triangle
  of the score matrix to `-inf` before softmax so the model can't peek at future characters.
- Softmax over the last dimension.
- Multiply the attention weights by V to get the output.

**Checkpoint:** print the shape of the scores matrix, and the matrix for one example (before
softmax) to confirm the upper triangle is `-inf`. This is the single most important shape in
the project — make sure you understand *why* it comes out the size it does.
- `Predict:` shape of the scores matrix = `__________`

### 5. Multi-head attention

- Run several heads in parallel (split `d_model` into `n_heads` pieces of size `head_dim`).
- Concatenate the head outputs back to `d_model`.
- One final linear projection.

**Checkpoint:** Attention doesn't change the shape, it mixes information across positions.
Confirm you understand why the shape is unchanged.
- `Predict:` output shape = `__________`  (compare it to the input shape)

### 6. The MLP

- Linear up to `4 * d_model`, a nonlinearity (GELU or ReLU), linear back down to `d_model`.

**Checkpoint:** input shape equals output shape. State why the middle is bigger.
- `Predict:` shape in the middle (after the up projection) = `__________`
- `Predict:` output shape = `__________`

### 7. The transformer block

Assemble one block with residual connections and layernorm:

- `x = x + attention(layernorm(x))`
- `x = x + mlp(layernorm(x))`

The residual (`x + ...`) is the part people gloss over. Sit on it for a second: the block
*adds* to the stream rather than replacing it. Know why that matters for gradients.

**Checkpoint:** A block is shape-preserving. Stack 2–3 of them — still shape-preserving.
- `Predict:` output shape of one block = `__________`
- `Predict:` output shape after stacking 3 = `__________`

### 8. The full model

- Embeddings → stack of blocks → final layernorm → a linear layer mapping `d_model` to
  `vocab_size` (the logits).

**Checkpoint:** for every position, a distribution over the 13 possible next characters.
- `Predict:` final output shape = `__________`  (this one's worth getting right before you run)

### 9. The training loop

- Loss: cross-entropy between logits and targets. Reshape logits to `(batch * seq_len,
  vocab_size)` and targets to `(batch * seq_len,)`.
- Optimizer: AdamW.
- Loop: get a batch, forward, compute loss, backward, step, zero grad. Print loss
  periodically.

**Checkpoint:** loss goes *down*. If it's flat or NaN, stop and debug before anything else —
this is where you learn the most. A NaN usually means the mask or a shape is wrong; a flat
loss usually means the targets are misaligned. Both are worth the hours they'll cost.
- `Predict:` shape of logits after reshaping = `__________`
- `Predict:` shape of targets after reshaping = `__________`

### 10. Generation + the real test

- Write a function: given a prompt string like `"12+37="`, feed it in, take the logits at
  the last position, pick the highest-probability next character, append it, repeat.
- Feed it held-out problems it never saw in training.

**Checkpoint (the whole point):** it correctly completes addition problems it was never
trained on. That's a transformer you built from scratch, generalizing. Done.

---

## Suggested starting hyperparameters

Small enough to train on CPU or a free Colab T4 in minutes. Don't tune these until it works.

- `d_model`: 64
- `n_heads`: 4
- `n_layers`: 3
- `seq_len`: whatever your padded length is (~8)
- batch size: 64
- learning rate: 3e-4
- steps: a few thousand; watch the loss

---

## What to look up vs. what to sit with

**Fine to look up:** exact PyTorch syntax (`nn.Embedding`, how to apply a causal mask,
cross-entropy argument shapes). Syntax isn't the lesson.

**Sit with it, don't look up:** why the score matrix is `seq_len × seq_len`; why the causal
mask is needed; why the target is the input shifted by one; why residuals help. If you look
these up you skip the actual learning. Struggle here is the product.

---

## When it works

You'll have built every piece of a transformer yourself and watched it generalize. The blur
is gone. *Then* pick up the roofline / scaling-book material — you'll have solid ground for
it. And Flash Attention will finally be exciting instead of opaque, because you'll know
exactly which matrix it's refusing to materialize: the `seq_len × seq_len` one from step 4.

Two natural extensions once the base works, both straight off Feinberg's list:
- Push to 3-digit numbers and see where it breaks.
- Derive Chinchilla-style scaling for this toy: vary model size and data, plot the curves.
