import argparse
from pathlib import Path

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
        description="Qwen3-TTS voice cloning (Base models).",
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
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--language", default=config["language"], help="Language or Auto")
    parser.add_argument(
        "--out",
        default=config["out"],
        help="Output wav path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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

    wavs, sr = model.generate_voice_clone(
        text=args.text,
        language=args.language,
        ref_audio=args.ref_audio,
        ref_text=ref_text,
        x_vector_only_mode=args.x_vector_only,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), wavs[0], sr)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
