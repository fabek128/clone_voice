import argparse

try:
    from huggingface_hub import snapshot_download
except Exception as exc:  # pragma: no cover - user environment dependent
    raise SystemExit(
        "huggingface_hub is required. Install dependencies with: pip install -r requirements.txt"
    ) from exc

from tts_config import load_config, merge_config


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
        description="Pre-download Qwen3-TTS model weights.",
        parents=[base],
    )
    parser.add_argument(
        "--model",
        default=config["model"],
        help="HF model id",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache dir",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional model revision",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    kwargs = {}
    if args.cache_dir:
        kwargs["cache_dir"] = args.cache_dir
    if args.revision:
        kwargs["revision"] = args.revision

    path = snapshot_download(repo_id=args.model, **kwargs)
    print(f"Downloaded to: {path}")


if __name__ == "__main__":
    main()
