"""Pydantic v2 configuration schema for VidwaanAI Fine-Tuning Pipeline."""

from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class LoraParameters(BaseModel):
    """Configuration for Low-Rank Adaptation (LoRA / QLoRA)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rank: int = Field(default=16, ge=1, le=128, description="LoRA rank dimension (r)")
    scale: float = Field(default=16.0, gt=0.0, le=256.0, description="LoRA scaling factor (alpha)")
    dropout: float = Field(
        default=0.0, ge=0.0, le=0.5, description="LoRA dropout rate (0 for Unsloth)"
    )
    target_modules: List[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        description="Transformer projection layers to adapt",
    )

    @field_validator("target_modules")
    @classmethod
    def validate_target_modules(cls, v: List[str]) -> List[str]:
        valid_modules = {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }
        for mod in v:
            if mod not in valid_modules:
                raise ValueError(
                    f"Invalid target module '{mod}'. Must be a subset of {sorted(valid_modules)}"
                )
        return v


class TrainingParameters(BaseModel):
    """Hyperparameters and runtime arguments for Supervised Fine-Tuning."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    batch_size: int = Field(default=2, ge=1, le=64, description="Per-device train batch size")
    grad_accumulation_steps: int = Field(
        default=4, ge=1, le=128, description="Gradient accumulation steps"
    )
    learning_rate: float = Field(default=2e-4, gt=0.0, le=1e-2, description="Peak learning rate")
    max_steps: Optional[int] = Field(
        default=None, ge=1, description="Max training steps (overrides epochs)"
    )
    num_train_epochs: Optional[int] = Field(
        default=3, ge=1, le=100, description="Number of training epochs"
    )
    warmup_steps: int = Field(default=5, ge=0, description="Linear warmup steps")
    weight_decay: float = Field(default=0.01, ge=0.0, le=0.5, description="AdamW weight decay")
    grad_checkpoint: bool = Field(default=True, description="Enable gradient checkpointing")
    random_seed: int = Field(default=3407, description="Reproducibility seed")
    output_dir: Path = Field(
        default=Path("./tmp_checkpoints"), description="Intermediate checkpoint path"
    )
    adapter_path: Path = Field(default=Path("./adapters"), description="Final adapter output path")
    logging_steps: int = Field(default=5, ge=1, description="Logging frequency in steps")
    save_steps: int = Field(default=50, ge=1, description="Checkpoint save frequency")
    eval_steps: int = Field(default=50, ge=1, description="Validation evaluation frequency")


class TrainingPipelineConfig(BaseModel):
    """Root configuration validator for VidwaanAI training pipeline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(
        default="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
        description="Base model identifier on Hugging Face Hub or local path",
    )
    max_seq_length: int = Field(
        default=2048, ge=256, le=8192, description="Maximum token sequence length"
    )
    load_in_4bit: bool = Field(default=True, description="Enforce 4-bit NF4 quantization (QLoRA)")
    data_dir: Path = Field(
        default=Path("./data"), description="Path to directory containing dataset files"
    )
    lora_parameters: LoraParameters = Field(default_factory=LoraParameters)
    training: TrainingParameters = Field(default_factory=TrainingParameters)

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> "TrainingPipelineConfig":
        """Load and validate pipeline configuration from a YAML file."""
        path = Path(yaml_path)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}

        # Backward compatibility for flat configs
        lora_raw = raw_data.get("lora_parameters", {})
        training_raw = raw_data.get("training", {})

        # Inherit top-level legacy keys if present
        if "batch_size" in raw_data and "batch_size" not in training_raw:
            training_raw["batch_size"] = raw_data["batch_size"]
        if "grad_accumulation_steps" in raw_data and "grad_accumulation_steps" not in training_raw:
            training_raw["grad_accumulation_steps"] = raw_data["grad_accumulation_steps"]
        if "learning_rate" in raw_data and "learning_rate" not in training_raw:
            training_raw["learning_rate"] = float(raw_data["learning_rate"])
        if "iters" in raw_data and "max_steps" not in training_raw:
            training_raw["max_steps"] = raw_data["iters"]
        if "adapter_path" in raw_data and "adapter_path" not in training_raw:
            training_raw["adapter_path"] = raw_data["adapter_path"]
        if "grad_checkpoint" in raw_data and "grad_checkpoint" not in training_raw:
            training_raw["grad_checkpoint"] = raw_data["grad_checkpoint"]

        clean_dict = {
            "model": raw_data.get("model", "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"),
            "max_seq_length": raw_data.get("max_seq_length", 2048),
            "load_in_4bit": raw_data.get("load_in_4bit", True),
            "data_dir": raw_data.get("data", raw_data.get("data_dir", "./data")),
            "lora_parameters": lora_raw,
            "training": training_raw,
        }

        return cls.model_validate(clean_dict)
