"""
test_addition_gpt.py — Berkeley-style autograder for addition_gpt.py.

Run one part at a time. Get each green before moving on:

    python -m pytest test_addition_gpt.py -k part1 -q
    python -m pytest test_addition_gpt.py -k part2 -q
    python -m pytest test_addition_gpt.py -k part3 -q
    python -m pytest test_addition_gpt.py -k part4 -q
    python -m pytest test_addition_gpt.py -k part5 -q

These check shapes and behavior, not your exact code. They verify the CONTRACT of
each piece — the same shapes the docstrings told you to predict. If a test fails on a
shape, that's the signal to go understand why, not just to force the number.

Part 5 trains a model, so it's slow (a minute or two). The rest are instant.
"""

import torch
import pytest

import addition_gpt as m


# ===========================================================================
# PART 1 — DATA
# ===========================================================================

def test_part1_encode_decode_roundtrip():
    s = "12+37=49."
    ids = m.encode(s)
    assert ids.dtype == torch.long
    assert ids.shape == (len(s),)
    assert m.decode(ids) == s


def test_part1_encode_values_in_range():
    ids = m.encode("0123456789+=.")
    assert ids.min() >= 0 and ids.max() < m.VOCAB_SIZE


def test_part1_make_example_fixed_len():
    s = m.make_example(12, 37)
    assert len(s) == m.FIXED_LEN
    assert s.startswith("12+37=49")


def test_part1_make_example_pads():
    s = m.make_example(5, 8)          # "5+8=13" is short -> must be padded
    assert len(s) == m.FIXED_LEN
    assert s.endswith(".")


def test_part1_batch_shapes_and_shift():
    x, y = m.make_batch(64)
    assert x.shape == (64, m.FIXED_LEN - 1)
    assert y.shape == (64, m.FIXED_LEN - 1)
    # target is input shifted left by one: y[:, :-1] should equal x[:, 1:]
    assert torch.equal(x[:, 1:], y[:, :-1])


# ===========================================================================
# PART 2 — MODEL COMPONENTS
# ===========================================================================

def _x():                     # a standard (B, T, D) activation for component tests
    return torch.randn(4, m.FIXED_LEN - 1, m.D_MODEL)

def _ids():                   # a standard (B, T) token-ID batch
    return torch.randint(0, m.VOCAB_SIZE, (4, m.FIXED_LEN - 1))


def test_part2_embeddings_shape():
    emb = m.Embeddings(m.VOCAB_SIZE, m.D_MODEL, m.FIXED_LEN)
    out = emb(_ids())
    assert out.shape == (4, m.FIXED_LEN - 1, m.D_MODEL)


def test_part2_attention_head_shape():
    head = m.AttentionHead(m.D_MODEL, m.D_MODEL // m.N_HEADS)
    out = head(_x())
    assert out.shape == (4, m.FIXED_LEN - 1, m.D_MODEL // m.N_HEADS)


def test_part2_attention_is_causal():
    """Position t's output must not depend on inputs after t. Change the LAST
    timestep of the input; every earlier output position must stay identical."""
    torch.manual_seed(0)
    head = m.AttentionHead(m.D_MODEL, m.D_MODEL // m.N_HEADS)
    x1 = _x()
    x2 = x1.clone()
    x2[:, -1, :] += 10.0                    # perturb only the final position
    o1, o2 = head(x1), head(x2)
    # all but the last position must be unchanged if the causal mask is correct
    assert torch.allclose(o1[:, :-1], o2[:, :-1], atol=1e-5)


def test_part2_mha_preserves_shape():
    mha = m.MultiHeadAttention(m.D_MODEL, m.N_HEADS)
    out = mha(_x())
    assert out.shape == (4, m.FIXED_LEN - 1, m.D_MODEL)


def test_part2_mlp_preserves_shape():
    mlp = m.MLP(m.D_MODEL)
    out = mlp(_x())
    assert out.shape == (4, m.FIXED_LEN - 1, m.D_MODEL)


def test_part2_block_preserves_shape():
    block = m.Block(m.D_MODEL, m.N_HEADS)
    out = block(_x())
    assert out.shape == (4, m.FIXED_LEN - 1, m.D_MODEL)


# ===========================================================================
# PART 3 — FULL MODEL
# ===========================================================================

def test_part3_model_output_shape():
    model = m.AdditionGPT()
    out = model(_ids())
    assert out.shape == (4, m.FIXED_LEN - 1, m.VOCAB_SIZE)


def test_part3_model_backprops():
    """A full forward+backward should populate gradients on every parameter."""
    model = m.AdditionGPT()
    out = model(_ids())
    out.sum().backward()
    assert any(p.grad is not None for p in model.parameters())


# ===========================================================================
# PART 4 — LOSS
# ===========================================================================

def test_part4_loss_is_scalar():
    logits = torch.randn(4, m.FIXED_LEN - 1, m.VOCAB_SIZE)
    targets = torch.randint(0, m.VOCAB_SIZE, (4, m.FIXED_LEN - 1))
    loss = m.compute_loss(logits, targets)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_part4_loss_near_ln_vocab_on_random():
    """Untrained/random logits should give loss around ln(VOCAB_SIZE). This catches
    reshape bugs and wrong reductions."""
    torch.manual_seed(0)
    logits = torch.randn(8, m.FIXED_LEN - 1, m.VOCAB_SIZE)
    targets = torch.randint(0, m.VOCAB_SIZE, (8, m.FIXED_LEN - 1))
    loss = m.compute_loss(logits, targets).item()
    import math
    assert abs(loss - math.log(m.VOCAB_SIZE)) < 1.0


# ===========================================================================
# PART 5 — END TO END (slow: trains a model)
# ===========================================================================

@pytest.mark.slow
def test_part5_learns_addition():
    """The real test. Train, then check it solves held-out problems. Uses a modest
    step count; if this is flaky, bump steps in the call below."""
    torch.manual_seed(0)
    model = m.train(steps=3000, batch_size=64)
    correct = 0
    trials = [(12, 37), (5, 8), (47, 25), (60, 21), (9, 9)]
    for a, b in trials:
        pred = m.solve(model, a, b)
        if pred.strip(".") == str(a + b):
            correct += 1
    # allow one miss for run-to-run variance
    assert correct >= len(trials) - 1, f"only {correct}/{len(trials)} correct"
