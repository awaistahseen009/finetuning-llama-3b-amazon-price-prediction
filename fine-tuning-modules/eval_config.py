# ===============================
# Eval Config Loader
# ===============================
import os
import yaml
from dotenv import load_dotenv

load_dotenv(override=True)

with open("eval_config.yaml", "r") as f:
    CFG = yaml.safe_load(f)

PROJECT_RUN_NAME = f"{CFG['project']['project_name']}-{CFG['run']['run_name']}"
HUB_MODEL_NAME = f"{CFG['project']['hf_user']}/{PROJECT_RUN_NAME}"
DATASET_NAME = f"{CFG['dataset']['user']}/{CFG['dataset']['name']}"

HF_TOKEN = os.getenv("HF_TOKEN")
