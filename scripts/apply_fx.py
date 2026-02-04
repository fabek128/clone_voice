import argparse
from pathlib import Path

from fx_utils import FxConfig, PRESETS, apply_fx_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply FX to a WAV/MP3 using Pedalboard + pydub.")
    parser.add_argument("--input", required=True, help="Input wav/mp3 path")
    parser.add_argument("--output", default=None, help="Output path (default: add _fx)")
    parser.add_argument(
        "--preset",
        default="none",
        choices=["none"] + sorted(PRESETS.keys()),
        help="Preset FX chain",
    )

    parser.add_argument("--gain-db", type=float, default=None)
    parser.add_argument("--pitch", type=float, default=None, help="Pitch shift in semitones")
    parser.add_argument("--reverb", type=float, default=None, help="Reverb amount 0..1")
    parser.add_argument("--echo-ms", type=float, default=None, help="Echo delay in ms")
    parser.add_argument("--echo-feedback", type=float, default=None, help="Echo feedback 0..1")
    parser.add_argument("--echo-mix", type=float, default=None, help="Echo mix 0..1")
    parser.add_argument("--distortion", type=float, default=None, help="Distortion drive dB")
    parser.add_argument("--lowpass-hz", type=float, default=None)
    parser.add_argument("--highpass-hz", type=float, default=None)

    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--compress-threshold", type=float, default=-24.0)
    parser.add_argument("--compress-ratio", type=float, default=3.0)
    parser.add_argument("--limit", action="store_true")
    parser.add_argument("--limit-threshold", type=float, default=-1.0)
    parser.add_argument("--normalize", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")

    cfg = FxConfig(
        preset=args.preset,
        gain_db=args.gain_db,
        pitch=args.pitch,
        reverb=args.reverb,
        echo_ms=args.echo_ms,
        echo_feedback=args.echo_feedback,
        echo_mix=args.echo_mix,
        distortion=args.distortion,
        lowpass_hz=args.lowpass_hz,
        highpass_hz=args.highpass_hz,
        compress=args.compress,
        compress_threshold=args.compress_threshold,
        compress_ratio=args.compress_ratio,
        limit=args.limit,
        limit_threshold=args.limit_threshold,
        normalize=args.normalize,
    )

    output_path = Path(args.output) if args.output else None
    output_path = apply_fx_file(input_path, output_path, cfg)
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
