import argparse
from pathlib import Path
from typing import List

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from tts_config import load_config, merge_config, resolve_device, resolve_dtype


def parse_args() -> argparse.Namespace:
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument(
        "--config",
        default="config.json",
        help="Path to config JSON (defaults shown in config.json)",
    )
    known, _ = base.parse_known_args()

    config = merge_config(load_config(known.config))

    parser = argparse.ArgumentParser(
        description="Qwen3-TTS batch voice cloning with cached prompts.",
        parents=[base],
    )
    parser.add_argument(
        "--model",
        default=config["model"],
        help="HF model id or local path",
    )
    parser.add_argument(
        "--device",
        default=config["device"],
        help="auto, cuda:0, cuda:1, or cpu",
    )
    parser.add_argument(
        "--dtype",
        default=config["dtype"],
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Torch dtype to use",
    )
    parser.add_argument(
        "--attn",
        default=config["attn"],
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        help="Attention implementation (optional)",
    )
    parser.add_argument("--ref-audio", required=True, help="Path or URL")
    parser.add_argument("--ref-text", default=None, help="Transcript for ref audio")
    parser.add_argument(
        "--x-vector-only",
        action="store_true",
        help="Use speaker embedding only (no ref_text)",
    )
    parser.add_argument(
        "--texts-file",
        default=None,
        help="Text file with one line per sentence",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=None,
        help="Single text item (can be repeated)",
    )
    parser.add_argument("--language", default=config["language"], help="Language or Auto")
    parser.add_argument(
        "--out-dir",
        default=config["batch_out_dir"],
        help="Output folder for wavs",
    )
    parser.add_argument(
        "--out-prefix",
        default="line",
        help="Prefix for output wav files",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(config["batch_chunk_size"]),
        help="Batch size for generation",
    )
    return parser.parse_args()


def read_texts_file(path: str) -> List[str]:
    content = Path(path).read_text(encoding="utf-8").splitlines()
    items: List[str] = []
    for line in content:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        items.append(line)
    return items


def collect_texts(texts_file: str | None, texts: List[str] | None) -> List[str]:
    items: List[str] = []
    if texts_file:
        items.extend(read_texts_file(texts_file))
    if texts:
        items.extend([t for t in texts if t and t.strip()])
    return items


def chunked(items: List[str], size: int) -> List[List[str]]:
    if size <= 0:
        raise SystemExit("chunk-size must be a positive integer")
    return [items[i : i + size] for i in range(0, len(items), size)]


def main() -> None:
    args = parse_args()

    texts = collect_texts(args.texts_file, args.text)
    if not texts:
        raise SystemExit("Provide --texts-file and/or at least one --text.")

    device = resolve_device(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested but torch.cuda.is_available() is False.")

    dtype = resolve_dtype(device, args.dtype)

    model_kwargs = {
        "pretrained_model_name_or_path": args.model,
        "device_map": device,
        "dtype": dtype,
    }
    if args.attn != "auto":
        model_kwargs["attn_implementation"] = args.attn

    model = Qwen3TTSModel.from_pretrained(**model_kwargs)

    if args.x_vector_only:
        ref_text = None
    else:
        if not args.ref_text:
            raise SystemExit("ref_text is required unless --x-vector-only is set.")
        ref_text = args.ref_text

    prompt_items = model.create_voice_clone_prompt(
        ref_audio=args.ref_audio,
        ref_text=ref_text,
        x_vector_only_mode=args.x_vector_only,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = 1
    for batch in chunked(texts, args.chunk_size):
        languages = [args.language] * len(batch)
        wavs, sr = model.generate_voice_clone(
            text=batch,
            language=languages,
            voice_clone_prompt=prompt_items,
        )
        for wav in wavs:
            out_path = out_dir / f"{args.out_prefix}_{index:04d}.wav"
            sf.write(str(out_path), wav, sr)
            print(f"Wrote: {out_path}")
            index += 1


if __name__ == "__main__":
    main()
