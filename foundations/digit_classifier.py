import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
        self.first_layer = nn.Linear(784, 512)
        self.second_layer = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.next_linear = nn.Linear(512, 10)
        self.output_layer = nn.Sigmoid()

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        x = self.first_layer(images)
        x = self.second_layer(x)
        x = self.dropout(x)
        x = self.next_linear(x)
        x = self.output_layer(x)
        return torch.round(x, decimals=4)
