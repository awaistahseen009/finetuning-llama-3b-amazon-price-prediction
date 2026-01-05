# ===============================
# Model & Tokenizer
# ===============================
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from config import CFG


def get_precision():
    capability = torch.cuda.get_device_capability()
    return capability[0] >= 8


def get_quant_config(use_bf16):
    if CFG["quantization"]["use_4bit"]:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    return BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        CFG["project"]["base_model"],
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model():
    use_bf16 = get_precision()
    quant_config = get_quant_config(use_bf16)

    model = AutoModelForCausalLM.from_pretrained(
        CFG["project"]["base_model"],
        quantization_config=quant_config,
        device_map="auto",
    )

    return model, use_bf16


def get_lora_config():
    r = CFG["lora"]["r"]

    return LoraConfig(
        r=r,
        lora_alpha=r * 2,
        lora_dropout=CFG["lora"]["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=CFG["lora"]["target_modules"],
    )
