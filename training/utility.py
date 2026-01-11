import tiktoken
import torch
from model.architecture import generate_text_simple

def text_to_token_ids(text, tokenizer):
    ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    ids = torch.tensor(ids).unsqueeze(0)
    return ids

def token_ids_to_text(ids, tokenizer):
    return tokenizer.decode(ids.squeeze(0).tolist())