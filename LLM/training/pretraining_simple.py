"""
Script for pretraining a small GPT-2 flash attention 124M parameter model
on books from Project Gutenberg.
"""

import argparse
import os
import sys
from pathlib import Path
import time
from datetime import datetime
import tiktoken
import torch
import wandb

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from dataloader.dataloader import Train_dataloader
from model.architecture import GPTModelFlashAttn
from trainer import Trainer
from model.dummy_model import load_yaml
from utility import plot_losses

config = load_yaml("../model/config/model_config_train.yaml")["model"]
tokenizer = tiktoken.get_encoding("gpt2")
trainer = Trainer()

def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    return text_data 


def create_dataloaders(text_data, train_ratio, batch_size, max_length, stride, num_workers=0):
    split_idx = int(train_ratio * len(text_data))

    train_loader = Train_dataloader(
        config=config,
        text=text_data[:split_idx],
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        tokenizer=tokenizer,
    )
    train_loader = train_loader.get_dataloader()

    val_loader = Train_dataloader(
        config=config,
        text=text_data[split_idx:],
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        tokenizer=tokenizer
    )
    val_loader = val_loader.get_dataloader()

    return train_loader, val_loader


def convert_time(seconds):
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return int(hours), int(minutes), int(seconds)


def print_eta(start_time, book_start_time, index, total_files):
    book_end_time = time.time()  # End time of processing this book
    elapsed_time = book_end_time - book_start_time
    total_elapsed_time = book_end_time - start_time
    books_remaining = total_files - index
    average_time_per_book = total_elapsed_time / index
    eta = average_time_per_book * books_remaining

    book_h, book_m, book_s = convert_time(elapsed_time)
    total_h, total_m, total_s = convert_time(total_elapsed_time)
    eta_h, eta_m, eta_s = convert_time(eta)

    print(f"Book processed in {book_h}h {book_m}m {book_s}s"
          f"\nTotal time elapsed {total_h}h {total_m}m {total_s}s"
          f"\nETA for remaining books: {eta_h}h {eta_m}m {eta_s}s")

def assign(left, right):
    if left.shape != right.shape:
            raise ValueError(f"Shape mismatch. Left: {left.shape}, "
                             "Right: {right.shape}"
                             )
    return torch.nn.Paramater(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


def train_model_simple(model, optimizer, device, n_epochs,
                       eval_freq, eval_iter, print_sample_iter, start_context,
                       output_dir, save_ckpt_freq, tokenizer,
                       batch_size=1024, train_ratio=0.90):

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1
    start_time = time.time()

    try:
        for epoch in range(n_epochs):

            # Iterate over the books in the training corpus
            # RESUMING TRAINING FROM BOOK 7
            for index, file_path in enumerate(all_files[6:], 1):
                book_start_time = time.time()
                text_data = read_text_file(file_path) + " <|endoftext|> "

                wandb.log({"current_book": index, "book_path": file_path})

                # TODO: pre-tokenize books 
                print(f"Tokenizing file {index} of {total_files}: {file_path}")

                # Initialize new data loaders for each book
                train_loader, val_loader = create_dataloaders(

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


def train_model_simple(model, optimizer, device, n_epochs,
                       eval_freq, eval_iter, print_sample_iter, start_context,
                       output_dir, save_ckpt_freq, tokenizer,
                       batch_size=1024, train_ratio=0.90):

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1
    start_time = time.time()

    try:
        for epoch in range(n_epochs):

            # Iterate over the books in the training corpus
            # RESUMING TRAINING FROM BOOK 7
            for index, file_path in enumerate(all_files[6:], 1):
                book_start_time = time.time()
                text_data = read_text_file(file_path) + " <|endoftext|> "

                wandb.log({"current_book": index, "book_path": file_path})

                # TODO: pre-tokenize books 
                print(f"Tokenizing file {index} of {total_files}: {file_path}")

                # Initialize new data loaders for each book
                train_loader, val_loader = create_dataloaders(
                    text_data,
                    train_ratio=train_ratio,
                    batch_size=batch_size,
                    max_length=config["context_length"],
                    stride=config["context_length"],
                    num_workers=0
                )
                print("Training ...")

                model.train()
                for input_batch, target_batch in train_loader:
                    optimizer.zero_grad()
                    loss = trainer.calc_loss_batch(input_batch, target_batch, model, device)
                    loss.backward()
                    optimizer.step()
                    tokens_seen += input_batch.numel()
                    global_step += 1

                    # wandb progress track log
                    wandb.log({
                        "train/loss_step": loss.item(),
                        "train/tokens_seen": tokens_seen,
                        "train/global_step": global_step,
    parser.add_argument("--print_sample_iter", type=int, default=1000,
                        help="Iterations between printing sample outputs")
    parser.add_argument("--eval_freq", type=int, default=100,
                        help="Frequency of evaluations during training")
    parser.add_argument("--save_ckpt_freq", type=int, default=100_000,
                        help="Frequency of saving model checkpoints during training")
    parser.add_argument("--lr", type=float, default=6e-4,
                        help="Learning rate for the optimizer")
    # Changed batch size to 16 here, smoother gradients + faster training?
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--chkpt_path", type=str, default=None,
                        help=".pth file containing model and optimizer state dict")
    parser.add_argument("--debug", type=bool, default=False,
                        help="Uses a very small model for debugging purposes")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123)
    model = GPTModelFlashAttn(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    
    # wandb project initialization
    wandb.init(
        project="gpt2-gutenberg-pretraining",
        config={
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.n_epochs,
            "model_params": config,
            "train_ratio": 0.90,
        },
        name=f"gpt2-124M-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    wandb.watch(model, log="all", log_freq=100)


    # if checkpoint file is included, load
    if args.chkpt_path:
        checkpoint = torch.load(args.chkpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        model.to(device)

    data_dir = args.data_dir
    all_files = [os.path.join(path, name) for path, subdirs, files
                 in os.walk(data_dir) for name in files if name.endswith((".txt"))]
    total_files = len(all_files)

    if total_files == 0:
        print("No training text files found. Make sure you "
              "selected the correct input directory")
        quit()
    print("Total files:", total_files)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, optimizer, device,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        eval_freq=args.eval_freq,
        eval_iter=1,
        print_sample_iter=args.print_sample_iter,
        output_dir=output_dir,
        save_ckpt_freq=args.save_ckpt_freq,
        start_context="The sky is looking",
        tokenizer=tokenizer
    )

    today = datetime.today().strftime('%Y-%m-%d %H-%M-%S')

    torch.save(model.state_dict(), output_dir / "model_pg_final.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        },
        f"model_and_optimizer_{today}.pth"
    )
    wandb.save(str(output_dir / "model_pg_final.pth"))

    epochs_tensor = torch.linspace(0, args.n_epochs, len(train_losses))
    fig = plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    wandb.log({"final_loss_plot": wandb.Image(fig)})
    
    wandb.finish()
    print(f"Maximum GPU memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
