import torch
from torch.utils.data import Dataset, DataLoader

"""
    Implementation of a data loader using torch dataset. Stack input and target by overlapping them offset
    by stride length. Used for training the LLM efficiently.
"""

class Train_dataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - max_length, stride):
            input = token_ids[i:max_length + i]
            target = token_ids[i + 1:i + max_length + 1]


            self.input_ids.append(torch.tensor(input))
            self.target_ids.append(torch.tensor(target))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

    
class Train_dataloader():
    def __init__(
        self,
        text,
        batch_size,
        max_length,
        stride,
        shuffle,
        drop_last,
        num_workers,
        tokenizer
    ):

        self.dataset = Train_dataset(text, tokenizer, max_length, stride)

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
        )

    def get_dataloader(self):
        return self.dataloader
