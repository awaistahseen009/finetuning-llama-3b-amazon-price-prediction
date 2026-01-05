# ===============================
# Model Loader
# ===============================
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from eval_config import CFG, HUB_MODEL_NAME


def get_precision():
    capability = torch.cuda.get_device_capability()
    return capability[0] >= 8


def get_quant_config(use_bf16):
    if CFG["quantization"]["use_4bit"]:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
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

    base_model = AutoModelForCausalLM.from_pretrained(
        CFG["project"]["base_model"],
        quantization_config=quant_config,
        device_map="auto",
    )

    base_model.generation_config.pad_token_id = base_model.config.eos_token_id

    model = PeftModel.from_pretrained(
        base_model,
        HUB_MODEL_NAME,
        revision=CFG["run"]["revision"],
    )

    return model, use_bf16
