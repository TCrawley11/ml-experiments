"""
Script for pretraining a small GPT-2 flash attention 124M parameter model
on books from Project Gutenberg.
"""

import argparse
import os
import sys
from pathlib import Path
import time
import datetime
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
            # NOTE* THIS IS TRAINING ON 25 BOOKS!
            for index, file_path in enumerate(all_files[:25], 1):
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
                        "train/epoch": epoch
                    })

                    # Optional evaluation step
                    if global_step % eval_freq == 0:
                        train_loss, val_loss = trainer.evaluate_model(
                            model, train_loader, val_loader, device, eval_iter)
                        train_losses.append(train_loss)
                        val_losses.append(val_loss)
                        track_tokens_seen.append(tokens_seen)

                        # wandb loss log
                        wandb.log({
                            "eval/train_loss": train_loss,
                            "eval/val_loss": val_loss
                        })
                        print(f"Ep {epoch+1} (Step {global_step}): "
                              f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

                    # Generate text passage
                    if global_step % print_sample_iter == 0:
                        trainer.generate_and_print_sample(
                            model, tokenizer, device, start_context
                        )

                if global_step % save_ckpt_freq:
                    file_name = output_dir / f"model_pg_{global_step}.pth"
                    torch.save(model.state_dict(), file_name)
                    wandb.save(str(file_name))
                    print(f"Saved {file_name}")

                book_elapsed = time.time() - book_start_time
                wandb.log({
                    "timing/book_processing_time": book_elapsed,
                    "timing/books_completed": index
                })

                print_eta(start_time, book_start_time, index, total_files)

    except KeyboardInterrupt:
        file_name = output_dir / f"model_pg_{global_step}_interrupted.pth"
        torch.save({
            model.state_dict(),
            optimizer.state_dict()
        },
            file_name)
        print(f"Saved {file_name}")
    
    finally:
        wandb.finish()

    return train_losses, val_losses, track_tokens_seen


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="GPT Model Training Configuration")

    parser.add_argument("--data_dir", type=str, default="gutenberg_preprocessed",
                        help="Directory containing the training data")
    parser.add_argument("--output_dir", type=str, default="model_checkpoints",
                        help="Directory where the model checkpoints will be saved")
    parser.add_argument("--n_epochs", type=int, default=1,
                        help="Number of epochs to train the model")
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
        name=f"gpt2-124M-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    wandb.watch(model, log="all", log_freq=100)


    # if checkpoint file is included, load
    if args.chkpt_path:
        checkpoint = torch.load(args.chkpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
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