import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gpt_download import download_and_load_gpt2
from generate_gpt2 import load_weights_into_gpt
from model.architecture import GPTModelFlashAttn
from training import trainer
import yaml
import tiktoken

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

trainer.generate_and_print_sample(
    model,
    tokenizer,
    device="cuda",
    start_context=START_CONTEXT)
