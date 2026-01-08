import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, emb_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # small constant that is added to variance to prevent divison by zero
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # GPT-2 approximation of gelu activation function found via curve fitting
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))