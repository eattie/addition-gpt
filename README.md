# addition-gpt

(REPO IS THE BLANK TEMPLATE SO ANYONE CAN DO THE PROJECT FOR FUN)

Build a GPT that learns integer addition, from scratch. A from-the-ground-up
transformer implementation structured as a fill-in-the-blanks project with a
test-based autograder.

Every function body in `addition_gpt.py` is left as `raise NotImplementedError`.
The docstrings specify what to build and the exact tensor shapes; the tests in
`test_addition_gpt.py` verify each part independently.

## Setup

```bash
pip install -r requirements.txt
```

## Working through it

Implement one part at a time, top to bottom, getting each green before moving on:

```bash
pytest -k part1 -q     # data
pytest -k part2 -q     # model components
pytest -k part3 -q     # full model
pytest -k part4 -q     # loss
pytest -m "not slow"   # everything except the end-to-end training test
pytest -k part5        # the real test: trains, then checks it can add (slow)
```

Once the model trains:

```bash
python addition_gpt.py
```

## The one rule

Each docstring has `Predict:` lines. Fill in the shape you expect **before** you
write the code. When your guess is wrong, stop and understand why before fixing it —
that gap is the point of the project.

See `SPEC.md` for the full walkthrough and the reasoning behind each part.
