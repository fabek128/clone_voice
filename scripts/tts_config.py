import json
from pathlib import Path
from typing import Any, Dict

import torch

DEFAULT_CONFIG: Dict[str, Any] = {
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "device": "auto",
    "dtype": "auto",
    "attn": "auto",
    "language": "Spanish",
    "out": "data/outputs/clone.wav",
    "batch_out_dir": "data/outputs/batch",
    "batch_chunk_size": 8,
    "ui_presets": "presets.json",
}

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.is_file():
        return {}
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {cfg_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Config at {cfg_path} must be a JSON object")
    return data


def merge_config(config: Dict[str, Any]) -> Dict[str, Any]:
    merged = DEFAULT_CONFIG.copy()
    for key, value in config.items():
        if value is not None:
            merged[key] = value
    return merged


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def resolve_dtype(device: str, dtype_arg: str) -> torch.dtype:
    if dtype_arg != "auto":
        return DTYPE_MAP[dtype_arg]
    if device.startswith("cuda"):
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32
