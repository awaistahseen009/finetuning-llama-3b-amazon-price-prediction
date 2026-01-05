# ===============================
# Evaluation Script
# ===============================
import torch
from datasets import load_dataset
from huggingface_hub import login
from transformers import set_seed

from eval_config import CFG, DATASET_NAME, HF_TOKEN
from eval_model import load_model, load_tokenizer
from util import evaluate


# ===============================
# Auth
# ===============================
login(HF_TOKEN, add_to_git_credential=True)


# ===============================
# Dataset
# ===============================
dataset = load_dataset(DATASET_NAME)
test_dataset = dataset["test"]


# ===============================
# Model
# ===============================
tokenizer = load_tokenizer()
model, _ = load_model()

print(f"Memory footprint: {model.get_memory_footprint() / 1e6:.1f} MB")


# ===============================
# Prediction Function
# ===============================
def model_predict(item):
    inputs = tokenizer(
        item["prompt"],
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=CFG["generation"]["max_new_tokens"],
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, prompt_len:]

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ===============================
# Run Evaluation
# ===============================
set_seed(CFG["generation"]["seed"])
evaluate(model_predict, test_dataset)
