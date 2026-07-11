# My GPT — Built from Scratch

> Assembled from the NeetCode ML course on [NeetCode.io](https://neetcode.io)  
> Built by **Krishna Kaushal** on July 11, 2026

Every file in this project is a solution from the NeetCode ML course.
The problems progressively build from gradient descent fundamentals all the way to a working GPT.

## Project Structure

```
model/          Attention, Transformer, GPT architecture
  attention.py             Self-attention head
  multi_head_attention.py  Multi-headed attention
  transformer.py           Transformer block
  gpt.py                   GPT model
  normalization.py         Layer normalization
  batch_normalization.py   Batch normalization
  rms_normalization.py     RMS normalization
  embeddings.py            Word embeddings
  positional_encoding.py   Positional encoding
  kv_cache.py              KV-Cache for fast inference
  grouped_query_attention.py  Grouped query attention

data/           Data pipeline
  tokenizer.py                BPE tokenizer
  vocab.py                    Character-level vocabulary
  loader.py                   Batched training data loader
  dataset.py                  GPT dataset preparation
  nlp_preprocessing.py        NLP preprocessing
  tokenizer_utils.py          Tokenization edge cases

train.py        GPT training loop (+ runnable demo)
generate.py     Text generation (+ runnable demo)

foundations/    Neural network primitives built from scratch
  gradient_descent.py, activations.py, softmax.py, loss.py,
  linear_regression.py, neuron.py, backprop.py, mlp.py, ...
```

## Quick Start

```bash
pip install -r requirements.txt
python train.py      # trains a tiny char-level GPT, writes gpt_checkpoint.pt
python generate.py   # samples text from the checkpoint
```

## Using course solutions

Each problem file exposes a `Solution` class (or `nn.Module` subclass) matching the NeetCode API:

```python
from foundations.gradient_descent import Solution as GD
from foundations.activations import Solution as Acts
from model.gpt import GPT
from data.vocab import Solution as Vocab

print(GD().get_minimizer(iterations=10, learning_rate=0.01, init=5))
```

Import from the specific module rather than `from foundations import *`, since many modules share the name `Solution`.

## Course

This project was built by completing the [NeetCode ML Course](https://neetcode.io/practice?tab=coreSkills&topic=Machine+Learning):

- Math Foundations (gradient descent, activations, loss functions)
- Neural Networks from scratch (neuron, backprop, MLP)
- PyTorch fundamentals
- NLP pipeline (embeddings, tokenization, attention)
- Transformer architecture
- GPT model + text generation
