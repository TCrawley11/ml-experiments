import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpt_download import download_and_load_gpt2
from generate_gpt2 import load_weights_into_gpt
from model.architecture import GPTModelFlashAttn
from training import trainer
from instruct_dataloader import make_dataloaders
import yaml
import tiktoken
import torch
import time

from model.dummy_model import load_yaml

MODEL_SIZE = "355M"
DEVICE = "cuda"
START_CONTEXT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Convert the active sentence to passive: 'The chef cooks the meal every day.'
"""

settings, params = download_and_load_gpt2(
    model_size=MODEL_SIZE,
    models_dir="gpt2"
    )

config = load_yaml("model/config/model_config_instruct_gpt2M.yaml")["model"]
model = GPTModelFlashAttn(config)
load_weights_into_gpt(model, params)
model.to(DEVICE)
#model.eval()
tokenizer = tiktoken.get_encoding("gpt2")
trainer = trainer.Trainer()

#trainer.generate_and_print_sample(
#    model,
#    tokenizer,
#    device="cuda",
#    start_context=START_CONTEXT)

train_loader, test_loader, eval_loader = make_dataloaders()

with torch.no_grad():
    train_loss = trainer.calc_loss_loader(
        train_loader,
        model,
        device='cuda',
        num_batches=5
    )

    eval_loss = trainer.calc_loss_loader(
        eval_loader,
            model,
            device='cuda',
            num_batches=5
    )

start_time = time.time()
optimizer = torch.optim.AdamW(
    model.parameters(),
        lr=0.00005,
        weight_decay=0.1
)
num_epochs = 2

train_losses, val_losses, tokens_seen = trainer.train_model_simple(
    model=model,
    train_loader=train_loader,
    val_loader=eval_loader,
    optimizer=optimizer,
    device='cuda',
    num_epochs=num_epochs,
    eval_freq=5,
    eval_iter=5,
    start_context=START_CONTEXT,
    tokenizer=tokenizer,
)

end_time = time.time()
exec_time_mins = (end_time - start_time) / 60
print(f"Training time comppleted in {exec_time_mins:.2f} minutes")

