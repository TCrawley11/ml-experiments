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

def collate_v1(batch, pad_token_id = 50256, device = "cpu"):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_list = []
    
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )

        # remove the extra padding added earlier
        inputs = torch.tensor(padded[:-1])
        inputs_list.append(inputs)

    inputs_tensor = torch.stack(inputs_list).to(device)
    return inputs_tensor


# adding targets allocation
def collate_v2(batch, pad_token_id = 50256, device = "cpu"):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_list = []
    targets_list = []
    
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )

        # remove the extra padding added earlier
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        inputs_list.append(inputs)
        targets_list.append(targets)

    inputs_tensor = torch.stack(inputs_list).to(device)
    targets_tesnsor = torch.stack(targets_list).to(device)
    return inputs_tensor, targets_tesnsor


# add ignore index to replace padded
def collate_fn(batch, ignore_index = -100, pad_token_id = 50256, allowed_max_length = None, device = "cpu"):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_list = []
    targets_list = []
    
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )

        # remove the extra padding added earlier
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_list.append(inputs)
        targets_list.append(targets)

    inputs_tensor = torch.stack(inputs_list).to(device)
    targets_tesnsor = torch.stack(targets_list).to(device)
    return inputs_tensor, targets_tesnsor

