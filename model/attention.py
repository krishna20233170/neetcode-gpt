import torch
import torch.nn as nn
from torchtyping import TensorType


class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.attention_dim = attention_dim
        self.key_proj = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query_proj = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value_proj = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask
        # 4. Softmax then scores @ V
        Q = self.query_proj(embedded)
        K = self.key_proj(embedded)
        V = self.value_proj(embedded)

        scores = (Q @ K.transpose(-2, -1)) / (self.attention_dim ** 0.5)

        context_length = embedded.shape[1]
        lower_triangular = torch.tril(torch.ones(context_length, context_length, device=embedded.device))
        mask = lower_triangular == 0
        scores = scores.masked_fill(mask, float("-inf"))
        scores = nn.functional.softmax(scores, dim=2)

        return torch.round(scores @ V, decimals=4)
