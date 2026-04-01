import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import tiktoken
from dataset import customized_collate_fn, InstructionDataset
from load_and_split import get_format_split

num_workers = 0 # to use cuda with multiprocessing I need to use the 'spawn' start method
batch_size = 2

def make_dataloaders():

    train_arr, test_arr, eval_arr = get_format_split()
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # train
    train_dataset = InstructionDataset(
        data=train_arr,
        tokenizer=tokenizer
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )
    
    # test
    test_dataset = InstructionDataset(
        data=test_arr,
        tokenizer=tokenizer
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )
    
    # evaluation
    eval_dataset = InstructionDataset(
        data=eval_arr,
        tokenizer=tokenizer
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    return train_loader, test_loader, eval_loader
