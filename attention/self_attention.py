import torch
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.w_query = nn.Parameter(torch.rand(d_in, d_out)) 
        self.w_key   = nn.Parameter(torch.rand(d_in, d_out)) 
        self.w_value = nn.Parameter(torch.rand(d_in, d_out)) 

    def forward(self, x):
        queries = x @ self.w_query 
        keys    = x @ self.w_key 
        values  = x @ self.w_value 

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vecs = attn_weights @ values
        return context_vecs

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    
    def forward(self, x):
        queries = self.w_query(x)
        keys    = self.w_key(x)
        values  = self.w_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vecs = attn_weights @ values
        return context_vecs
