import modal
import re
from fastapi import FastAPI
from pydantic import BaseModel

app = modal.App("llama-price-predictor-2", include_source=True)

secrets = modal.Secret.from_name("openai-api-key")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-runtime-ubuntu22.04",
        add_python="3.10",
    )
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "peft",
        "accelerate",
        "trl",
        "bitsandbytes",
        "huggingface_hub",
        "litellm",
        "openai",
        "fastapi",
        "pydantic>=2.0.0",
        "pyyaml",
    )
    .add_local_python_source("inference")
    .add_local_python_source("eval_model")
    .add_local_python_source("evaluate")
    .add_local_python_source("train")
    .add_local_python_source("config")
    .add_local_python_source("eval_config")
    .add_local_file(local_path="config.yaml", remote_path="/root/config.yaml")
    .add_local_file(local_path="eval_config.yaml", remote_path="/root/eval_config.yaml")
)

volume = modal.Volume.from_name("hf-cache", create_if_missing=True)

SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""

class PredictRequest(BaseModel):
    content: str

class PredictResponse(BaseModel):
    price: float

def extract_price(text: str) -> float:
    match = re.search(
        r"[-+]?\d*\.\d+|\d+",
        text.replace("$", "").replace(",", ""),
    )
    if match:
        return float(match.group())
    return 999.0

@app.cls(
    gpu="A10G",
    image=image,
    volumes={"/root/.cache/huggingface": volume},
    secrets=[secrets],
    timeout=600,
)
class ModelService:

    @modal.method()
    def predict(self, content: str) -> float:
        import torch
        from litellm import completion
        from inference import Predictor

        predictor = Predictor()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        rewrite = completion(
            model="openai/gpt-4o",
            messages=messages,
            temperature=0.0,
            max_tokens=256,
        )

        rewritten_description = rewrite.choices[0].message.content.strip()

        print(f"GPT-4o FULL RESPONSE:\n{rewritten_description}")

        # Directly prepend the price question to the full GPT response
        final_prompt = f"What is the price of the product, rounded to the nearest dollar?\n\n{rewritten_description}"

        print(f"FINAL PROMPT SENT TO LOCAL MODEL:\n{final_prompt}")

        raw_output = predictor.predict(final_prompt)

        print(f"LOCAL MODEL RAW OUTPUT: '{raw_output}'")

        cleaned_text = raw_output.replace("$", "").replace(",", "").strip()
        print(f"CLEANED TEXT (before regex): '{cleaned_text}'")

        price = extract_price(raw_output)
        print(f"EXTRACTED PRICE (after regex): {price}")

        final_price = price if price > 0 else 999.0
        print(f"FINAL PRICE RETURNED: {final_price}")

        return final_price

web_app = FastAPI()

@web_app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    model = ModelService()
    price = await model.predict.remote.aio(request.content)
    return {"price": price}

@app.function()
@modal.asgi_app()
def fastapi_app():
    return web_app