"""GPT training loop (NeetCode problem) + runnable char-level demo.

Usage:
    python train.py
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.vocab import Solution as Vocab
from model.gpt import GPT


class TrainableGPT(GPT):
    """Same architecture as the course GPT, but without rounding logits.

    NeetCode's ``GPT.forward`` rounds to 4 decimals for judge output. That is
    correct for the problem API but kills gradient signal, so the runnable demo
    uses this subclass instead.
    """

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        torch.manual_seed(0)
        embedded = self.word_embeddings(context)
        positions = torch.arange(context.shape[1], device=context.device)
        embedded = embedded + self.position_embeddings(positions)
        output = self.final_norm(self.transformer_blocks(embedded))
        return self.vocab_projection(output)


class Solution:
    def train(
        self,
        model: nn.Module,
        data: torch.Tensor,
        epochs: int,
        context_length: int,
        batch_size: int,
        lr: float,
    ) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        loss = torch.tensor(0.0)

        for epoch in range(epochs):
            torch.manual_seed(epoch)

            indices = torch.randint(len(data) - context_length, (batch_size,)).tolist()
            X = torch.stack([data[i : i + context_length] for i in indices]).long()
            Y = torch.stack([data[i + 1 : i + 1 + context_length] for i in indices]).long()

            logits = model(X)  # (batch_size, context_length, vocab_size)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.reshape(B * T, C), Y.reshape(B * T))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return round(float(loss.item()), 4)


# ---------------------------------------------------------------------------
# Runnable demo: train a tiny char-level GPT and save checkpoint
# ---------------------------------------------------------------------------

DEFAULT_TEXT = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life.
"""

CHECKPOINT_PATH = "gpt_checkpoint.pt"


def main() -> None:
    text = DEFAULT_TEXT.strip()
    vocab = Vocab()
    stoi, itos = vocab.build_vocab(text)
    data = torch.tensor(vocab.encode(text, stoi), dtype=torch.long)

    context_length = 16
    model_dim = 32
    num_blocks = 2
    num_heads = 4
    batch_size = 8
    epochs = 200
    lr = 1e-2

    model = TrainableGPT(
        vocab_size=len(stoi),
        context_length=context_length,
        model_dim=model_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
    )

    print(f"Vocab size: {len(stoi)} | Tokens: {len(data)} | Params: {sum(p.numel() for p in model.parameters())}")
    print(f"Training for {epochs} epochs...")

    final_loss = Solution().train(
        model=model,
        data=data,
        epochs=epochs,
        context_length=context_length,
        batch_size=batch_size,
        lr=lr,
    )
    print(f"Final loss: {final_loss}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "stoi": stoi,
            "itos": itos,
            "config": {
                "vocab_size": len(stoi),
                "context_length": context_length,
                "model_dim": model_dim,
                "num_blocks": num_blocks,
                "num_heads": num_heads,
            },
        },
        CHECKPOINT_PATH,
    )
    print(f"Saved checkpoint to {os.path.abspath(CHECKPOINT_PATH)}")


if __name__ == "__main__":
    main()
