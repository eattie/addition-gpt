"""
addition_gpt.py — Build a GPT that learns addition, from scratch.

You implement every body below. Nothing here is filled in for you: the docstrings
tell you WHAT each function must do and the shapes it must produce; test_addition_gpt.py
tells you WHEN each part is right. Work top to bottom. Do not start a part until the
previous part's tests pass.

    python -m pytest test_addition_gpt.py -k part1   # then part2, part3, ...

WORKFLOW FOR EVERY FUNCTION:
  1. Read the docstring. Find the "Predict:" line(s).
  2. Write your predicted shape in the blank BEFORE coding. Actually type it in.
  3. Implement the body.
  4. Run that part's tests. When a shape is wrong, stop and understand why before fixing.

Stack: PyTorch. Everything runs on CPU in seconds except training (a couple minutes).

Vocab (13 tokens): digits 0-9, then '+', '=', and a pad token '.'.
Example string: "12+37=49." padded to FIXED_LEN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config — don't tune until the whole thing works end to end.
# ---------------------------------------------------------------------------
VOCAB = "0123456789+=."      # index 12 ('.') is the pad token
PAD_ID = VOCAB.index(".")
VOCAB_SIZE = len(VOCAB)
FIXED_LEN = 9                # "12+37=49." -> 9 chars; all examples padded to this

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 3
LR = 3e-4


# ===========================================================================
# PART 1 — DATA
# ===========================================================================

def encode(s: str) -> torch.Tensor:
    """Map a string to a LongTensor of token IDs using VOCAB.

    Predict: shape for input of length L = __________

    Args:
        s: a string, every char of which is in VOCAB.
    Returns:
        LongTensor of shape (len(s),).
    """
    raise NotImplementedError


def decode(t: torch.Tensor) -> str:
    """Inverse of encode: map a 1-D LongTensor of IDs back to a string."""
    raise NotImplementedError


def make_example(a: int, b: int) -> str:
    """Format one addition example as a fixed-length string.

    "12+37=49" then pad with '.' up to FIXED_LEN. E.g. (12, 37) -> "12+37=49.".

    Returns:
        A string of length exactly FIXED_LEN.
    """
    raise NotImplementedError


def make_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of (input, target) pairs for next-char prediction.

    Draw random a, b in [0, 99]. Build each example string, encode it, then split
    into input x = chars[:-1] and target y = chars[1:] (target is input shifted
    left by one).

    Predict: shape of x = __________
    Predict: shape of y = __________

    Returns:
        (x, y), each a LongTensor of shape (batch_size, FIXED_LEN - 1).
    """
    raise NotImplementedError


# ===========================================================================
# PART 2 — MODEL COMPONENTS
# Build each as its own module and test in isolation before assembling.
# ===========================================================================

class Embeddings(nn.Module):
    """Token embedding + learned position embedding, summed.

    Predict: output shape for input (B, T) = __________
    """

    def __init__(self, vocab_size: int, d_model: int, max_len: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T) token IDs -> (B, T, d_model)."""
        raise NotImplementedError


class AttentionHead(nn.Module):
    """A single causal self-attention head.

    Q, K, V projections; scores = Q @ K^T / sqrt(head_dim); causal mask (upper
    triangle -> -inf) BEFORE softmax; softmax over last dim; weighted sum with V.

    Predict: shape of the scores matrix for input (B, T, d_model) = __________
             (this is THE shape of the project — know why before you run it)

    Args at init:
        d_model, head_dim.
    """

    def __init__(self, d_model: int, head_dim: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, head_dim)."""
        raise NotImplementedError


class MultiHeadAttention(nn.Module):
    """n_heads AttentionHeads in parallel, concatenated, then a final projection.

    Predict: output shape for input (B, T, d_model) = __________
             (compare to the input shape — should it change?)
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model)."""
        raise NotImplementedError


class MLP(nn.Module):
    """Position-wise feed-forward: Linear up to 4*d_model, GELU, Linear back down.

    Predict: shape after the up-projection = __________
    """

    def __init__(self, d_model: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model)."""
        raise NotImplementedError


class Block(nn.Module):
    """One transformer block with pre-norm and residual connections:

        x = x + attn(layernorm(x))
        x = x + mlp(layernorm(x))

    Note it ADDS to x rather than replacing it. Know why that matters for gradients.
    Predict: output shape for input (B, T, d_model) = __________
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ===========================================================================
# PART 3 — THE FULL MODEL
# ===========================================================================

class AdditionGPT(nn.Module):
    """Embeddings -> N Blocks -> final LayerNorm -> Linear to vocab logits.

    Predict: output shape for input (B, T) = __________
    """

    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T) token IDs -> (B, T, VOCAB_SIZE) logits."""
        raise NotImplementedError


# ===========================================================================
# PART 4 — TRAINING
# ===========================================================================

def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over all positions.

    Reshape logits (B, T, V) -> (B*T, V) and targets (B, T) -> (B*T,), then use
    F.cross_entropy. (Simplest version: compute loss on every position, padding
    included. Optional upgrade later: ignore_index=PAD_ID.)

    Predict: shape of logits after reshape = __________
    Predict: shape of targets after reshape = __________

    Returns:
        A scalar loss tensor.
    """
    raise NotImplementedError


def train(steps: int = 3000, batch_size: int = 64) -> "AdditionGPT":
    """Standard loop: make_batch -> forward -> compute_loss -> backward -> step ->
    zero_grad. Use AdamW at LR. Print the loss every ~200 steps.

    If loss is flat or NaN, STOP and debug — that's the highest-value moment in the
    project. NaN usually = mask/shape bug; flat loss usually = misaligned targets.

    Returns:
        The trained model.
    """
    raise NotImplementedError


# ===========================================================================
# PART 5 — GENERATION (the real test)
# ===========================================================================

@torch.no_grad()
def solve(model: "AdditionGPT", a: int, b: int) -> str:
    """Autoregressively complete "a+b=" and return the digits the model predicts.

    Build the prompt "a+b=", encode it, and repeatedly: forward, take the logits at
    the LAST position, argmax to get the next token, append it, repeat until you hit
    FIXED_LEN or generate the pad token. Return the predicted answer as a string.

    This is the whole point: run it on pairs the model never trained on and see if
    it actually adds.
    """
    raise NotImplementedError


if __name__ == "__main__":
    model = train()
    for a, b in [(12, 37), (5, 8), (47, 25), (99, 1)]:
        print(f"{a}+{b}= {solve(model, a, b)}  (true {a + b})")
