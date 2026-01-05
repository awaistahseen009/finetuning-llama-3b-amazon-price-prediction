# ===============================
# Config Loader
# ===============================
import os
import yaml
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=True)

with open("config.yaml", "r") as f:
    CFG = yaml.safe_load(f)

RUN_NAME = f"{datetime.now():%Y-%m-%d_%H.%M.%S}"

PROJECT_RUN_NAME = f"{CFG['project']['project_name']}-{RUN_NAME}"
HUB_MODEL_NAME = f"{CFG['project']['username']}/{PROJECT_RUN_NAME}"

DATASET_NAME = f"{CFG['dataset']['user']}/{CFG['dataset']['name']}"

HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
