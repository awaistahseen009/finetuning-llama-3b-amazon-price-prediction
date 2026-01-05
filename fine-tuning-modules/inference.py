# ===============================
# Inference Logic
# ===============================
import torch
from transformers import set_seed
from eval_model import load_model, load_tokenizer
from eval_config import CFG


class Predictor:
    def __init__(self):
        self.tokenizer = load_tokenizer()
        self.model, _ = load_model()
        self.model.eval()
        set_seed(CFG["generation"]["seed"])

    def predict(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to("cuda")

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=CFG["generation"]["max_new_tokens"],
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, prompt_len:]

        return self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )
