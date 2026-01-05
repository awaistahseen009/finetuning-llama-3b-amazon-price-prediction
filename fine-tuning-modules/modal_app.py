import modal
import re
from fastapi import FastAPI
from pydantic import BaseModel

app = modal.App("llama-price-predictor-1", include_source=True)

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

        with torch.inference_mode():
            rewrite = completion(
                model="openai/gpt-4o",
                messages=messages,
                temperature=0.0,
                max_tokens=256,
            )

        raw_text = rewrite.choices[0].message.content.strip()

        lines = []
        for line in raw_text.split("\n"):
            if ": " in line:
                _, value = line.split(": ", 1)
                lines.append(value.strip())

        rewritten_prompt = lines[3] if len(lines) > 3 else raw_text

        with torch.inference_mode():
            raw_output = predictor.predict(rewritten_prompt)

        price = extract_price(raw_output)
        return price if price > 0 else 999.0

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