"""Central configuration for the Unintended Memorization empirical study."""

import os
import random
from pathlib import Path

import numpy as np

# --- Model & Reproducibility ---
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 42
DTYPE = "float32"

# --- Corpus & Background Prose ---
NUM_RECORDS = 256
RECORD_LENGTH = 96
BACKGROUND_DATASET = "Salesforce/wikitext"
BACKGROUND_CONFIG = "wikitext-103-raw-v1"
BACKGROUND_SPLIT = "train"
MIN_BG_TOKENS = 130
SHUFFLE_BACKGROUND = True       # False -> Run 1 (sequential), True -> Run 2 (shuffled)
SHUFFLE_BUFFER_SIZE = 10_000

# --- LoRA (Attention + MLP) ---
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# --- Training ---
NUM_EPOCHS = 2
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 1
LEARNING_RATE = 2e-4
LR_SCHEDULER = "constant"
WARMUP_STEPS = 0
WEIGHT_DECAY = 0.0

# --- Evaluation ---
MAX_NEW_TOKENS = 24

# --- Output Paths ---
ARTIFACTS_DIR = Path("/content/drive/MyDrive/unintended_memorization_artifacts")

CORPUS_DIR = ARTIFACTS_DIR / "corpus"
ADAPTER_DIR = ARTIFACTS_DIR / "lora_adapter"
RESULTS_DIR = ARTIFACTS_DIR / "results"
PLOTS_DIR = ARTIFACTS_DIR / "plots"


def ensure_dirs() -> None:
    for folder in (CORPUS_DIR, ADAPTER_DIR, RESULTS_DIR, PLOTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_torch_dtype():
    import torch

    dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtypes[DTYPE]
