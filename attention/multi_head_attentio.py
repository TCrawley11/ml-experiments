import torch

class MutiHeadAttention(torch.nn.Module):
    def __init__(
            self, d_in, d_out, context_length, num_heads, dropout, qkv_bias=False
    ): 
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by number of heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.w_query = torch.nn.Linear(d_in, d_out, qkv_bias)
        self.w_key = torch.nn.Linear(d_in, d_out, qkv_bias)
        self.w_value = torch.nn.Linear(d_in, d_out, qkv_bias)

        self.out_proj = torch.nn.Linear(d_out, d_out)
        self.dropout = torch.nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    
    def forward(self, input):
        b, num_tokens, d_in = input.shape
        queries = self.w_query(input)
        keys    = self.w_key(input)
        values   = self.w_value(input)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys    = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values  = values.view(b, num_tokens, self.num_heads, self.head_dim)

        queries = queries.transpose(1, 2)
        keys    = keys.transpose(1, 2)
        values  = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(
            attn_scores // keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )
        context_vec = self.out_proj(context_vec)
        return context_vec