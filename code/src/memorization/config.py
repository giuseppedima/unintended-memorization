"""Central experiment configuration loader backed by `.env`."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from memorization.exceptions import ConfigurationError

try:
    from dotenv import load_dotenv
except ImportError:

    def load_dotenv(
        dotenv_path: str | Path | None = None,
        override: bool = False,
    ) -> bool:
        """Minimal fallback `.env` parser when `python-dotenv` is unavailable."""
        if dotenv_path is None:
            return False
        path = Path(dotenv_path)
        if not path.exists():
            return False
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if override or key.strip() not in os.environ:
                os.environ[key.strip()] = value.strip()
        return True


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable configuration object for the entire empirical study."""

    # Model & Reproducibility
    model_name: str
    seed: int
    dtype: str

    # Corpus & Canary Protocol
    num_records: int
    record_length_tokens: int
    vocab_size: int
    num_canaries: int
    canaries_per_group: int
    secret_word_count: int

    # Background Source
    background_dataset: str
    background_dataset_config: str
    background_dataset_split: str
    min_bg_tokens: int

    # LoRA
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: tuple[str, ...]

    # Training
    num_epochs: int
    batch_size: int
    grad_accum_steps: int
    learning_rate: float
    lr_scheduler_type: str
    warmup_steps: int
    weight_decay: float

    # Evaluation
    max_new_tokens: int

    # Storage
    artifacts_dir: Path
    use_google_drive: bool
    drive_artifacts_dir: Path

    @property
    def active_artifacts_dir(self) -> Path:
        return self.drive_artifacts_dir if self.use_google_drive else self.artifacts_dir

    @property
    def corpus_dir(self) -> Path:
        return self.active_artifacts_dir / "corpus"

    @property
    def adapter_dir(self) -> Path:
        return self.active_artifacts_dir / "lora_adapter"

    @property
    def results_dir(self) -> Path:
        return self.active_artifacts_dir / "results"

    @property
    def plots_dir(self) -> Path:
        return self.active_artifacts_dir / "plots"

    def ensure_dirs(self) -> None:
        """Create all output artifact directories if missing."""
        for directory in (
            self.active_artifacts_dir,
            self.corpus_dir,
            self.adapter_dir,
            self.results_dir,
            self.plots_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def _str_to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _validate_protocol_config(cfg: ExperimentConfig) -> None:
    """Validate mandatory experimental protocol constraints."""
    protocol_checks = [
        (cfg.num_records == 256, "Protocol requires NUM_RECORDS == 256."),
        (cfg.record_length_tokens == 96, "Protocol requires RECORD_LENGTH_TOKENS == 96."),
        (cfg.vocab_size == 32, "Protocol requires VOCAB_SIZE == 32."),
        (cfg.num_canaries == 24, "Protocol requires NUM_CANARIES == 24."),
        (cfg.canaries_per_group == 6, "Protocol requires CANARIES_PER_GROUP == 6."),
        (cfg.secret_word_count == 4, "Protocol requires SECRET_WORD_COUNT == 4."),
        (cfg.batch_size > 0, "BATCH_SIZE must be a positive integer."),
        (
            cfg.num_records % cfg.batch_size == 0,
            f"BATCH_SIZE ({cfg.batch_size}) must divide NUM_RECORDS ({cfg.num_records}).",
        ),
        (
            (cfg.num_records // cfg.batch_size) >= 20,
            f"Mini-batches per epoch ({cfg.num_records // cfg.batch_size}) must be >= 20 "
            "to enforce Controlled Shuffling for Group 3 (20 copies).",
        ),
        (cfg.learning_rate > 0, "LEARNING_RATE must be strictly positive."),
        (cfg.num_epochs >= 1, "NUM_EPOCHS must be >= 1."),
        (len(cfg.lora_target_modules) > 0, "LORA_TARGET_MODULES cannot be empty."),
    ]
    for ok, message in protocol_checks:
        if not ok:
            raise ConfigurationError(message)


def load_config(env_path: str | Path | None = ".env") -> ExperimentConfig:
    """Load and validate configuration from a `.env` file and environment variables."""
    if env_path is not None and Path(env_path).exists():
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        load_dotenv(override=False)

    raw_modules = os.getenv(
        "LORA_TARGET_MODULES",
        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    target_modules = tuple(m.strip() for m in raw_modules.split(",") if m.strip())

    cfg = ExperimentConfig(
        model_name=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
        seed=int(os.getenv("SEED", "42")),
        dtype=os.getenv("DTYPE", "float32").strip().lower(),
        num_records=int(os.getenv("NUM_RECORDS", "256")),
        record_length_tokens=int(os.getenv("RECORD_LENGTH_TOKENS", "96")),
        vocab_size=int(os.getenv("VOCAB_SIZE", "32")),
        num_canaries=int(os.getenv("NUM_CANARIES", "24")),
        canaries_per_group=int(os.getenv("CANARIES_PER_GROUP", "6")),
        secret_word_count=int(os.getenv("SECRET_WORD_COUNT", "4")),
        background_dataset=os.getenv("BACKGROUND_DATASET", "Salesforce/wikitext"),
        background_dataset_config=os.getenv(
            "BACKGROUND_DATASET_CONFIG", "wikitext-103-raw-v1"
        ),
        background_dataset_split=os.getenv("BACKGROUND_DATASET_SPLIT", "train"),
        min_bg_tokens=int(os.getenv("MIN_BG_TOKENS", "130")),
        lora_r=int(os.getenv("LORA_R", "16")),
        lora_alpha=int(os.getenv("LORA_ALPHA", "32")),
        lora_dropout=float(os.getenv("LORA_DROPOUT", "0.0")),
        lora_target_modules=target_modules,
        num_epochs=int(os.getenv("NUM_EPOCHS", "2")),
        batch_size=int(os.getenv("BATCH_SIZE", "4")),
        grad_accum_steps=int(os.getenv("GRAD_ACCUM_STEPS", "1")),
        learning_rate=float(os.getenv("LEARNING_RATE", "2e-4")),
        lr_scheduler_type=os.getenv("LR_SCHEDULER_TYPE", "constant").strip().lower(),
        warmup_steps=int(os.getenv("WARMUP_STEPS", "0")),
        weight_decay=float(os.getenv("WEIGHT_DECAY", "0.0")),
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "24")),
        artifacts_dir=Path(os.getenv("ARTIFACTS_DIR", "artifacts")),
        use_google_drive=_str_to_bool(os.getenv("USE_GOOGLE_DRIVE", "false")),
        drive_artifacts_dir=Path(
            os.getenv(
                "DRIVE_ARTIFACTS_DIR",
                "/content/drive/MyDrive/unintended_memorization_artifacts",
            )
        ),
    )

    _validate_protocol_config(cfg)
    return cfg


def set_global_seed(seed: int) -> None:
    """Fix pseudo-random seeds across Python, NumPy, and PyTorch (when installed)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_torch_dtype(dtype_name: str):
    """Map string dtype identifier to `torch.dtype`."""
    import torch

    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ConfigurationError(
            f"Unsupported DTYPE '{dtype_name}'. Choose from {sorted(mapping)}."
        )
    return mapping[dtype_name]
