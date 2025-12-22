import torch
import torch.nn as nn
import yaml

class DummyGPTModel(nn.Module):
    def __init__(self, cfg_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = self.load_yaml(cfg_path)

        self.tok_emb  = nn.Linear(self.cfg["vocab_size"], self.cfg["emb_dim"]) 
        self.pos_emb  = nn.Linear(self.cfg["context_length"], self.cfg["emb_dim"]) 
        self.drop_emb = nn.Dropout(self.cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(self.cfg)
              for _ in range(self.cfg["n_layers"])]
        )
        self.final_norm = DummyLayerNorm(self.cfg["emb_dim"])
        self.out_head = nn.Linear(
            self.cfg["emb_dim"], self.cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds        
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    @staticmethod
    def load_yaml(path: str):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return config


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return x

class DummyLayerNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        return x