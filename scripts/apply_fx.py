import argparse
import inspect
from pathlib import Path

import numpy as np
from pydub import AudioSegment, effects as pydub_effects
from pedalboard import (
    Compressor,
    Delay,
    Distortion,
    Gain,
    HighpassFilter,
    Limiter,
    LowpassFilter,
    Pedalboard,
    PitchShift,
    Reverb,
)

PRESETS = {
    "thick": {
        "pitch": -2.0,
        "gain_db": 2.0,
        "lowpass_hz": 9000.0,
    },
    "reverb": {
        "reverb": 0.5,
    },
    "echo": {
        "echo_ms": 220.0,
        "echo_feedback": 0.35,
        "echo_mix": 0.25,
    },
    "broadcast": {
        "compress": True,
        "limit": True,
        "highpass_hz": 80.0,
        "gain_db": 3.0,
    },
}


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


def apply_preset_defaults(args: argparse.Namespace) -> None:
    preset = PRESETS.get(args.preset)
    if not preset:
        return
    for key, value in preset.items():
        current = getattr(args, key, None)
        if isinstance(value, bool):
            if current is False:
                setattr(args, key, value)
        else:
            if current is None:
                setattr(args, key, value)


def safe_effect(cls, **kwargs):
    sig = inspect.signature(cls.__init__)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters and v is not None}
    return cls(**filtered)


class AudioLoadError(SystemExit):
    pass


def load_audio(path: Path):
    try:
        segment = AudioSegment.from_file(path)
    except Exception as exc:
        raise AudioLoadError(
            "Failed to load audio. If input is mp3/m4a, ensure ffmpeg is installed and in PATH."
        ) from exc

    samples = np.array(segment.get_array_of_samples())
    if segment.channels > 1:
        samples = samples.reshape((-1, segment.channels))
    else:
        samples = samples.reshape((-1, 1))

    max_val = float(1 << (8 * segment.sample_width - 1))
    audio = samples.astype(np.float32) / max_val
    return audio, segment.frame_rate, segment.channels


def numpy_to_segment(audio: np.ndarray, sample_rate: int, channels: int) -> AudioSegment:
    audio = np.clip(audio, -1.0, 1.0)
    samples = (audio * 32767.0).astype(np.int16)
    raw = samples.tobytes()
    return AudioSegment(
        data=raw,
        sample_width=2,
        frame_rate=sample_rate,
        channels=channels,
    )


def build_board(args: argparse.Namespace) -> Pedalboard:
    board = Pedalboard()

    if args.highpass_hz:
        board.append(safe_effect(HighpassFilter, cutoff_frequency_hz=args.highpass_hz))
    if args.lowpass_hz:
        board.append(safe_effect(LowpassFilter, cutoff_frequency_hz=args.lowpass_hz))
    if args.pitch:
        board.append(safe_effect(PitchShift, semitones=args.pitch))
    if args.gain_db:
        board.append(safe_effect(Gain, gain_db=args.gain_db))
    if args.reverb:
        board.append(
            safe_effect(
                Reverb,
                room_size=args.reverb,
                wet_level=min(0.6, max(0.1, args.reverb)),
                dry_level=0.7,
            )
        )
    if args.echo_ms:
        feedback = args.echo_feedback if args.echo_feedback is not None else 0.35
        mix = args.echo_mix if args.echo_mix is not None else 0.25
        board.append(
            safe_effect(
                Delay,
                delay_seconds=args.echo_ms / 1000.0,
                feedback=feedback,
                mix=mix,
            )
        )
    if args.distortion:
        board.append(safe_effect(Distortion, drive_db=args.distortion))
    if args.compress:
        board.append(
            safe_effect(
                Compressor,
                threshold_db=args.compress_threshold,
                ratio=args.compress_ratio,
            )
        )
    if args.limit:
        board.append(safe_effect(Limiter, threshold_db=args.limit_threshold))

    return board


def resolve_output_path(input_path: Path, output_arg: str | None) -> Path:
    if output_arg:
        return Path(output_arg)
    return input_path.with_name(f"{input_path.stem}_fx{input_path.suffix}")


def main() -> None:
    args = parse_args()
    apply_preset_defaults(args)

    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")

    audio, sample_rate, channels = load_audio(input_path)
    board = build_board(args)
    if len(board):
        audio = board(audio, sample_rate)

    segment = numpy_to_segment(audio, sample_rate, channels)
    if args.normalize:
        segment = pydub_effects.normalize(segment)

    output_path = resolve_output_path(input_path, args.output)
    fmt = output_path.suffix.lstrip(".").lower() or "wav"
    segment.export(output_path, format=fmt)
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
