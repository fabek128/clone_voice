from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any, Dict, Optional

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

PRESETS: Dict[str, Dict[str, Any]] = {
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
    "clean": {
        "compress": True,
        "limit": True,
        "highpass_hz": 80.0,
        "lowpass_hz": 12000.0,
    },
}


@dataclass
class FxConfig:
    preset: str = "none"
    gain_db: Optional[float] = None
    pitch: Optional[float] = None
    reverb: Optional[float] = None
    echo_ms: Optional[float] = None
    echo_feedback: Optional[float] = None
    echo_mix: Optional[float] = None
    distortion: Optional[float] = None
    lowpass_hz: Optional[float] = None
    highpass_hz: Optional[float] = None

    compress: bool = False
    compress_threshold: float = -24.0
    compress_ratio: float = 3.0
    limit: bool = False
    limit_threshold: float = -1.0
    normalize: bool = False


def config_from_dict(data: Dict[str, Any]) -> FxConfig:
    config = FxConfig()
    for key, value in data.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def apply_preset_defaults(config: FxConfig) -> None:
    preset = PRESETS.get(config.preset)
    if not preset:
        return
    for key, value in preset.items():
        current = getattr(config, key, None)
        if isinstance(value, bool):
            if current is False:
                setattr(config, key, value)
        else:
            if current is None:
                setattr(config, key, value)


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


def build_board(config: FxConfig) -> Pedalboard:
    board = Pedalboard()

    if config.highpass_hz:
        board.append(safe_effect(HighpassFilter, cutoff_frequency_hz=config.highpass_hz))
    if config.lowpass_hz:
        board.append(safe_effect(LowpassFilter, cutoff_frequency_hz=config.lowpass_hz))
    if config.pitch:
        board.append(safe_effect(PitchShift, semitones=config.pitch))
    if config.gain_db:
        board.append(safe_effect(Gain, gain_db=config.gain_db))
    if config.reverb:
        board.append(
            safe_effect(
                Reverb,
                room_size=config.reverb,
                wet_level=min(0.6, max(0.1, config.reverb)),
                dry_level=0.7,
            )
        )
    if config.echo_ms:
        feedback = config.echo_feedback if config.echo_feedback is not None else 0.35
        mix = config.echo_mix if config.echo_mix is not None else 0.25
        board.append(
            safe_effect(
                Delay,
                delay_seconds=config.echo_ms / 1000.0,
                feedback=feedback,
                mix=mix,
            )
        )
    if config.distortion:
        board.append(safe_effect(Distortion, drive_db=config.distortion))
    if config.compress:
        board.append(
            safe_effect(
                Compressor,
                threshold_db=config.compress_threshold,
                ratio=config.compress_ratio,
            )
        )
    if config.limit:
        board.append(safe_effect(Limiter, threshold_db=config.limit_threshold))

    return board


def resolve_output_path(input_path: Path, output_arg: Optional[str]) -> Path:
    if output_arg:
        return Path(output_arg)
    return input_path.with_name(f"{input_path.stem}_fx{input_path.suffix}")


def apply_fx_array(audio: np.ndarray, sample_rate: int, config: FxConfig) -> np.ndarray:
    apply_preset_defaults(config)
    board = build_board(config)
    if len(board):
        audio = board(audio, sample_rate)
    return audio


def apply_fx_file(
    input_path: Path,
    output_path: Optional[Path],
    config: FxConfig,
) -> Path:
    apply_preset_defaults(config)
    audio, sample_rate, channels = load_audio(input_path)
    board = build_board(config)
    if len(board):
        audio = board(audio, sample_rate)

    segment = numpy_to_segment(audio, sample_rate, channels)
    if config.normalize:
        segment = pydub_effects.normalize(segment)

    output_path = resolve_output_path(input_path, str(output_path) if output_path else None)
    fmt = output_path.suffix.lstrip(".").lower() or "wav"
    segment.export(output_path, format=fmt)
    return output_path
