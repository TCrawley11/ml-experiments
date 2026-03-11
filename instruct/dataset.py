import torch
from torch.utils.data import Dataset

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_text = []

        for entry in data:
            self.encoded_text.append(tokenizer.encode(entry))


    def __getitem__(self, index):
        return self.encoded_text[index]

    def __len__(self):
        return len(self.encoded_text)
