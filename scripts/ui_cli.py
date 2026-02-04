import json
import subprocess
import sys
from pathlib import Path

from tts_config import load_config, merge_config

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"


def load_presets(path: str) -> list[dict]:
    preset_path = (ROOT / path).resolve() if path else None
    if not preset_path or not preset_path.is_file():
        return []
    try:
        data = json.loads(preset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def prompt_str(label: str, default: str | None = None, required: bool = False) -> str:
    while True:
        suffix = f" [{default}]" if default else ""
        value = input(f"{label}{suffix}: ").strip()
        if value:
            return value
        if default is not None:
            return default
        if not required:
            return ""


def prompt_yes_no(label: str, default: bool = False) -> bool:
    default_str = "y" if default else "n"
    value = input(f"{label} [y/n] (default {default_str}): ").strip().lower()
    if not value:
        return default
    return value in {"y", "yes"}


def choose_option(title: str, options: list[str]) -> int:
    print("\n" + title)
    for i, opt in enumerate(options, start=1):
        print(f"  {i}. {opt}")
    while True:
        value = input("Choose an option: ").strip()
        if value.isdigit():
            idx = int(value)
            if 1 <= idx <= len(options):
                return idx - 1


def run_cmd(args: list[str]) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def pick_text(presets: list[dict]) -> str:
    if not presets:
        return prompt_str("Text", required=True)
    options = [item.get("name", "preset") for item in presets] + ["Custom"]
    idx = choose_option("Text presets", options)
    if idx == len(options) - 1:
        return prompt_str("Text", required=True)
    return str(presets[idx].get("text", "")).strip()


def build_fx_args() -> list[str]:
    options = [
        "None",
        "Thick (darker)",
        "Reverb",
        "Echo",
        "Broadcast (compress/limit)",
        "Clean (filter/comp)",
        "Custom",
    ]
    idx = choose_option("FX preset", options)
    if idx == 0:
        return []

    if idx == 6:
        args: list[str] = []
        pitch = prompt_str("Pitch semitones (e.g. -2)")
        if pitch:
            args += ["--pitch", pitch]
        gain = prompt_str("Gain dB (e.g. 2)")
        if gain:
            args += ["--gain-db", gain]
        reverb = prompt_str("Reverb amount 0-1")
        if reverb:
            args += ["--reverb", reverb]
        echo_ms = prompt_str("Echo delay ms")
        if echo_ms:
            args += ["--echo-ms", echo_ms]
            echo_feedback = prompt_str("Echo feedback 0-1", default="0.35")
            echo_mix = prompt_str("Echo mix 0-1", default="0.25")
            args += ["--echo-feedback", echo_feedback, "--echo-mix", echo_mix]
        distortion = prompt_str("Distortion drive dB")
        if distortion:
            args += ["--distortion", distortion]
        highpass = prompt_str("Highpass Hz")
        if highpass:
            args += ["--highpass-hz", highpass]
        lowpass = prompt_str("Lowpass Hz")
        if lowpass:
            args += ["--lowpass-hz", lowpass]
        if prompt_yes_no("Apply compressor", default=False):
            args += ["--compress"]
        if prompt_yes_no("Apply limiter", default=False):
            args += ["--limit"]
        if prompt_yes_no("Normalize output", default=False):
            args += ["--normalize"]
        return args

    preset_map = {1: "thick", 2: "reverb", 3: "echo", 4: "broadcast", 5: "clean"}
    args = ["--preset", preset_map[idx]]
    if prompt_yes_no("Normalize output", default=False):
        args += ["--normalize"]
    return args


def apply_fx_to_file(input_path: str, output_path: str | None, fx_args: list[str]) -> None:
    cmd = [
        sys.executable,
        str(SCRIPTS / "apply_fx.py"),
        "--input",
        input_path,
    ]
    if output_path:
        cmd += ["--output", output_path]
    cmd += fx_args
    run_cmd(cmd)


def apply_fx_to_dir(directory: Path, fx_args: list[str], out_dir: Path | None) -> None:
    out_dir = out_dir or directory
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list(directory.glob("*.wav")) + list(directory.glob("*.mp3")) + list(directory.glob("*.m4a"))
    if not files:
        print("No audio files found in output folder.")
        return

    for path in files:
        out_path = out_dir / f"{path.stem}_fx{path.suffix}"
        apply_fx_to_file(str(path), str(out_path), fx_args)


def clone_single(config: dict, presets: list[dict]) -> None:
    ref_audio = prompt_str("Reference audio path", required=True)
    ref_text = prompt_str("Reference transcript (optional)")
    x_vector_only = False
    if not ref_text:
        x_vector_only = prompt_yes_no("Use x-vector-only mode", default=True)
    text = pick_text(presets)
    language = prompt_str("Language", default=config["language"])
    out_path = prompt_str("Output wav path", default=config["out"])

    cmd = [
        sys.executable,
        str(SCRIPTS / "clone_voice.py"),
        "--ref-audio",
        ref_audio,
        "--text",
        text,
        "--language",
        language,
        "--out",
        out_path,
    ]
    if ref_text:
        cmd += ["--ref-text", ref_text]
    if x_vector_only:
        cmd += ["--x-vector-only"]

    run_cmd(cmd)

    if prompt_yes_no("Apply FX to generated audio", default=False):
        fx_args = build_fx_args()
        if fx_args:
            apply_fx_to_file(out_path, None, fx_args)


def clone_batch(config: dict) -> None:
    ref_audio = prompt_str("Reference audio path", required=True)
    ref_text = prompt_str("Reference transcript (optional)")
    x_vector_only = False
    if not ref_text:
        x_vector_only = prompt_yes_no("Use x-vector-only mode", default=True)
    texts_file = prompt_str("Texts file path", required=True)
    language = prompt_str("Language", default=config["language"])
    out_dir = prompt_str("Output folder", default=config["batch_out_dir"])
    chunk_size = prompt_str("Chunk size", default=str(config["batch_chunk_size"]))

    cmd = [
        sys.executable,
        str(SCRIPTS / "batch_clone.py"),
        "--ref-audio",
        ref_audio,
        "--texts-file",
        texts_file,
        "--language",
        language,
        "--out-dir",
        out_dir,
        "--chunk-size",
        chunk_size,
    ]
    if ref_text:
        cmd += ["--ref-text", ref_text]
    if x_vector_only:
        cmd += ["--x-vector-only"]

    run_cmd(cmd)

    if prompt_yes_no("Apply FX to all outputs", default=False):
        fx_args = build_fx_args()
        if fx_args:
            fx_out_dir = prompt_str("FX output folder", default=out_dir)
            apply_fx_to_dir(Path(out_dir), fx_args, Path(fx_out_dir))


def apply_fx_flow() -> None:
    input_path = prompt_str("Input wav/mp3 path", required=True)
    output_path = prompt_str("Output path (optional)")
    fx_args = build_fx_args()
    if not fx_args:
        print("No FX selected.")
        return
    apply_fx_to_file(input_path, output_path or None, fx_args)


def download_model(config: dict) -> None:
    model = prompt_str("Model", default=config["model"])
    cache_dir = prompt_str("Cache dir (optional)")

    cmd = [
        sys.executable,
        str(SCRIPTS / "download_model.py"),
        "--model",
        model,
    ]
    if cache_dir:
        cmd += ["--cache-dir", cache_dir]

    run_cmd(cmd)


def main() -> None:
    config = merge_config(load_config("config.json"))
    presets = load_presets(config.get("ui_presets", "presets.json"))

    options = [
        "Clone single",
        "Batch clone",
        "Download model",
        "Apply FX to file",
        "Exit",
    ]

    while True:
        choice = choose_option("Qwen3-TTS CLI", options)
        try:
            if choice == 0:
                clone_single(config, presets)
            elif choice == 1:
                clone_batch(config)
            elif choice == 2:
                download_model(config)
            elif choice == 3:
                apply_fx_flow()
            else:
                return
        except subprocess.CalledProcessError as exc:
            print(f"Command failed: {exc}")


if __name__ == "__main__":
    main()
