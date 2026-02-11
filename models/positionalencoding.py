import math

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, is_sin=True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.is_sin = True

        position = torch.arange(max_len).unsqueeze(1)
        if is_sin:
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe)
        else:
            self.pe = nn.Parameter(torch.randn((max_len, 1, d_model)), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """

        x = x.transpose(1, 0)
        x = x + self.pe[: x.size(0)]
        x = x.transpose(1, 0)
        return self.dropout(x)


if __name__ == "__main__":
    pos_enc = PositionalEncoding(d_model=64, max_len=14)
