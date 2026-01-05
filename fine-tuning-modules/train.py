# ===============================
# Training Script
# ===============================
import os
import wandb
from datasets import load_dataset
from huggingface_hub import login
from trl import SFTTrainer, SFTConfig

from config import (
    CFG,
    RUN_NAME,
    PROJECT_RUN_NAME,
    HUB_MODEL_NAME,
    DATASET_NAME,
    HF_TOKEN,
    WANDB_API_KEY,
)
from model import load_model, load_tokenizer, get_lora_config


# ===============================
# Auth & Logging
# ===============================
login(HF_TOKEN, add_to_git_credential=True)

if CFG["logging"]["use_wandb"]:
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    wandb.init(
        project=CFG["project"]["project_name"],
        name=RUN_NAME,
    )


# ===============================
# Dataset
# ===============================
dataset = load_dataset(DATASET_NAME)

train_dataset = dataset["train"]
val_dataset = dataset["val"].select(range(CFG["dataset"]["val_size"]))


# ===============================
# Model Setup
# ===============================
tokenizer = load_tokenizer()
model, use_bf16 = load_model()
model.generation_config.pad_token_id = tokenizer.pad_token_id

print(f"Memory footprint: {model.get_memory_footprint() / 1e9:.1f} GB")


# ===============================
# Trainer Config
# ===============================
trainer_config = SFTConfig(
    output_dir=PROJECT_RUN_NAME,
    num_train_epochs=CFG["training"]["epochs"],
    per_device_train_batch_size=CFG["training"]["batch_size"],
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=CFG["training"]["gradient_accumulation_steps"],
    optim=CFG["training"]["optimizer"],
    save_steps=CFG["training"]["save_steps"],
    save_total_limit=10,
    logging_steps=CFG["training"]["log_steps"],
    learning_rate=CFG["training"]["learning_rate"],
    weight_decay=CFG["training"]["weight_decay"],
    fp16=not use_bf16,
    bf16=use_bf16,
    warmup_ratio=CFG["training"]["warmup_ratio"],
    lr_scheduler_type=CFG["training"]["lr_scheduler_type"],
    max_length=CFG["training"]["max_sequence_length"],
    group_by_length=True,
    report_to="wandb" if CFG["logging"]["use_wandb"] else None,
    run_name=RUN_NAME,
    push_to_hub=True,
    hub_model_id=HUB_MODEL_NAME,
    hub_private_repo=True,
    hub_strategy="every_save",
    eval_strategy="steps",
    eval_steps=CFG["training"]["save_steps"],
)


# ===============================
# Training
# ===============================
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=get_lora_config(),
    args=trainer_config,
)

trainer.train()
trainer.model.push_to_hub(PROJECT_RUN_NAME, private=True)

print(f"Saved to the hub: {PROJECT_RUN_NAME}")

if CFG["logging"]["use_wandb"]:
    wandb.finish()
