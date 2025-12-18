import torch

class Causal_attention(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.w_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key   = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )


    def forward(self, input):
        b, num_tokens, d_in = input.shape
        # calculate q, k, v 
        queries = self.w_query(input)
        keys    = self.w_key(input)
        values  = self.w_value(input)

        # calculate scores + apply mask
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        # Get weights and do a dropout layer
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # get the context vecs
        context_vec = attn_weights @ values

        return context_vec
