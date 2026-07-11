"""Text generation (NeetCode problem) + runnable demo from a trained checkpoint.

Usage:
    python train.py      # train first
    python generate.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
from torchtyping import TensorType

from train import TrainableGPT

CHECKPOINT_PATH = "gpt_checkpoint.pt"


class Solution:
    def generate(
        self,
        model,
        new_chars: int,
        context: TensorType[int],
        context_length: int,
        int_to_char: dict,
    ) -> str:
        generator = torch.manual_seed(0)
        initial_state = generator.get_state()
        result = []
        for _ in range(new_chars):
            # Crop context to max length the model can handle
            if context.shape[1] > context_length:
                context = context[:, -context_length:]

            # Forward pass -> logits for every position
            logits = model(context)  # (1, T, vocab_size)
            last_logits = logits[:, -1, :]  # (1, vocab_size)
            probs = nn.functional.softmax(last_logits, dim=-1)

            # Sample next token and reset RNG for reproducibility
            next_token = torch.multinomial(probs, 1, generator=generator)
            generator.set_state(initial_state)

            # Append token to context and decode
            context = torch.cat((context, next_token), dim=-1)
            result.append(int_to_char[next_token.item()])
        return "".join(result)


@torch.no_grad()
def sample_text(
    model: nn.Module,
    prompt_ids: list[int],
    context_length: int,
    int_to_char: dict,
    new_chars: int = 200,
    temperature: float = 0.8,
    seed: int = 42,
) -> str:
    """Demo sampler with advancing RNG (unlike the course Solution, which
    resets the generator every step for judge reproducibility)."""
    generator = torch.Generator().manual_seed(seed)
    context = torch.tensor([prompt_ids], dtype=torch.long)
    pieces: list[str] = []

    for _ in range(new_chars):
        if context.shape[1] > context_length:
            context = context[:, -context_length:]
        logits = model(context)[:, -1, :] / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1, generator=generator)
        context = torch.cat((context, next_token), dim=-1)
        pieces.append(int_to_char[next_token.item()])

    return "".join(pieces)


def main() -> None:
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"No checkpoint at {CHECKPOINT_PATH}. Run `python train.py` first.")
        sys.exit(1)

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    itos = {int(k): v for k, v in ckpt["itos"].items()}
    stoi = ckpt["stoi"]

    model = TrainableGPT(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        model_dim=config["model_dim"],
        num_blocks=config["num_blocks"],
        num_heads=config["num_heads"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    prompt = "To be"
    # Fall back to first vocab char if a prompt char is missing
    ids = [stoi.get(c, 0) for c in prompt]

    print(f"Prompt: {prompt!r}")
    generated = sample_text(
        model=model,
        prompt_ids=ids,
        context_length=config["context_length"],
        int_to_char=itos,
        new_chars=200,
        temperature=0.8,
    )
    print(f"Generated:\n{prompt}{generated}")


if __name__ == "__main__":
    main()
