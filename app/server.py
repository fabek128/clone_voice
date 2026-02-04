import asyncio
import json
import queue
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from qwen_tts import Qwen3TTSModel

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SCRIPTS_DIR))

from fx_utils import FxConfig, PRESETS, apply_fx_file, config_from_dict, numpy_to_segment  # noqa: E402
from tts_config import load_config, merge_config, resolve_device, resolve_dtype  # noqa: E402

DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "inputs" / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
HISTORY_PATH = DATA_DIR / "history.jsonl"
WEB_DIR = ROOT / "web"
STALE_SECONDS = 20 * 60

DEFAULTS = merge_config(load_config(str(ROOT / "config.json")))

app = FastAPI(title="Qwen3-TTS Manager", version="0.1.0")

app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def safe_suffix(name: str) -> str:
    suffix = Path(name).suffix.lower()
    if suffix in {".wav", ".mp3", ".m4a", ".flac", ".ogg"}:
        return suffix
    return ".wav"


def ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def write_audio(output_path: Path, audio: Any, sample_rate: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".wav":
        sf.write(str(output_path), audio, sample_rate)
        return

    segment = numpy_to_segment(audio.reshape((-1, 1)), sample_rate, 1)
    fmt = output_path.suffix.lstrip(".").lower() or "wav"
    segment.export(output_path, format=fmt)


def chunked(items: List[str], size: int) -> List[List[str]]:
    if size <= 0:
        raise ValueError("chunk_size must be positive")
    return [items[i : i + size] for i in range(0, len(items), size)]


class ModelCache:
    def __init__(self) -> None:
        self._cache: Dict[str, Qwen3TTSModel] = {}
        self._lock = threading.Lock()

    def get(self, model_id: str, device: str, dtype: torch.dtype, attn: str) -> Qwen3TTSModel:
        key = f"{model_id}|{device}|{dtype}|{attn}"
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            kwargs = {
                "pretrained_model_name_or_path": model_id,
                "device_map": device,
                "dtype": dtype,
            }
            if attn != "auto":
                kwargs["attn_implementation"] = attn
            model = Qwen3TTSModel.from_pretrained(**kwargs)
            self._cache[key] = model
            return model


MODEL_CACHE = ModelCache()


class TaskManager:
    def __init__(self) -> None:
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.order: List[str] = []
        self.lock = threading.Lock()
        self.queue: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.monitor = threading.Thread(target=self._monitor, daemon=True)
        self.worker.start()
        self.monitor.start()
        self._load_history()

    def _load_history(self) -> None:
        if not HISTORY_PATH.is_file():
            return
        try:
            lines = HISTORY_PATH.read_text(encoding="utf-8").splitlines()
        except Exception:
            return
        for line in lines[-200:]:
            try:
                task = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_id = task.get("id")
            if not task_id:
                continue
            self.tasks[task_id] = task
            self.order.append(task_id)

    def _append_history(self, task: Dict[str, Any]) -> None:
        try:
            HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            with HISTORY_PATH.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(task, ensure_ascii=True) + "\n")
        except Exception:
            pass

    def create(self, task_type: str, params: Dict[str, Any], func) -> Dict[str, Any]:
        task_id = uuid.uuid4().hex[:12]
        task = {
            "id": task_id,
            "type": task_type,
            "status": "queued",
            "message": "queued",
            "created_at": now_iso(),
            "started_at": None,
            "finished_at": None,
            "updated_at": now_iso(),
            "params": params,
            "outputs": [],
            "error": None,
            "error_detail": None,
        }
        with self.lock:
            self.tasks[task_id] = task
            self.order.insert(0, task_id)
        self.queue.put((task_id, func))
        return task

    def update(self, task_id: str, **updates: Any) -> None:
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                return
            updates["updated_at"] = now_iso()
            task.update(updates)

    def snapshot(self) -> List[Dict[str, Any]]:
        with self.lock:
            return [self.tasks[task_id] for task_id in self.order if task_id in self.tasks]

    def _worker(self) -> None:
        while True:
            task_id, func = self.queue.get()
            self.update(task_id, status="running", message="running", started_at=now_iso())
            try:
                outputs = func(task_id, self.update)
                self.update(
                    task_id,
                    status="done",
                    message="done",
                    finished_at=now_iso(),
                    outputs=outputs or [],
                )
            except Exception as exc:
                self.update(
                    task_id,
                    status="error",
                    message="error",
                    finished_at=now_iso(),
                    error=str(exc),
                    error_detail=traceback.format_exc(),
                )
            task = self.tasks.get(task_id)
            if task:
                self._append_history(task)

    def _monitor(self) -> None:
        while True:
            time.sleep(15)
            with self.lock:
                for task in self.tasks.values():
                    if task.get("status") != "running":
                        continue
                    updated_at = task.get("updated_at")
                    if not updated_at:
                        continue
                    try:
                        last = time.strptime(updated_at, "%Y-%m-%dT%H:%M:%S")
                    except ValueError:
                        continue
                    elapsed = time.time() - time.mktime(last)
                    if elapsed > STALE_SECONDS:
                        task["status"] = "stalled"
                        task["message"] = f"stalled > {STALE_SECONDS // 60}m"


TASKS = TaskManager()


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/config")
def get_config():
    return JSONResponse(
        {
            "defaults": DEFAULTS,
            "fx_presets": sorted(PRESETS.keys()),
        }
    )


@app.get("/api/tasks")
def list_tasks():
    return JSONResponse({"tasks": TASKS.snapshot()})


@app.get("/api/events")
async def events():
    async def event_stream():
        while True:
            payload = json.dumps({"tasks": TASKS.snapshot()}, ensure_ascii=True)
            yield f"data: {payload}\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/clone")
async def clone_voice(
    ref_audio: UploadFile = File(...),
    ref_text: str | None = Form(None),
    text: str | None = Form(None),
    texts: str | None = Form(None),
    mode: str = Form("single"),
    language: str = Form("Auto"),
    model: str = Form(DEFAULTS["model"]),
    device: str = Form(DEFAULTS["device"]),
    dtype: str = Form(DEFAULTS["dtype"]),
    attn: str = Form(DEFAULTS["attn"]),
    x_vector_only: str = Form("false"),
    output_format: str = Form("wav"),
    apply_fx: str = Form("false"),
    fx_config: str | None = Form(None),
    extra_kwargs: str | None = Form(None),
    chunk_size: int = Form(8),
    out_prefix: str = Form("line"),
    join_outputs: str = Form("false"),
):
    ensure_dirs()
    suffix = safe_suffix(ref_audio.filename or "ref.wav")
    task_dir = UPLOAD_DIR / uuid.uuid4().hex[:10]
    task_dir.mkdir(parents=True, exist_ok=True)
    ref_path = task_dir / f"ref{suffix}"
    with ref_path.open("wb") as handle:
        handle.write(await ref_audio.read())

    params = {
        "mode": mode,
        "language": language,
        "model": model,
        "device": device,
        "dtype": dtype,
        "attn": attn,
        "x_vector_only": parse_bool(x_vector_only),
        "output_format": output_format,
        "apply_fx": parse_bool(apply_fx),
        "join_outputs": parse_bool(join_outputs),
    }

    def run(task_id: str, update):
        update(task_id, message="loading model")
        resolved_device = resolve_device(device)
        resolved_dtype = resolve_dtype(resolved_device, dtype)
        if resolved_device.startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is False.")

        model_obj = MODEL_CACHE.get(model, resolved_device, resolved_dtype, attn)

        if parse_bool(x_vector_only):
            ref_text_local = None
        else:
            if not ref_text:
                raise SystemExit("ref_text is required unless x_vector_only is set.")
            ref_text_local = ref_text

        extra: Dict[str, Any] = {}
        if extra_kwargs:
            try:
                extra = json.loads(extra_kwargs)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid extra_kwargs JSON: {exc}") from exc
            if not isinstance(extra, dict):
                raise SystemExit("extra_kwargs must be a JSON object")

        forbidden = {
            "text",
            "language",
            "ref_audio",
            "ref_text",
            "x_vector_only_mode",
            "voice_clone_prompt",
        }
        extra = {k: v for k, v in extra.items() if k not in forbidden}

        output_dir = OUTPUT_DIR / task_id
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs: List[Dict[str, Any]] = []

        if mode == "batch":
            if not texts:
                raise SystemExit("Batch mode requires texts (one per line).")
            items = [line.strip() for line in texts.splitlines() if line.strip()]
            if not items:
                raise SystemExit("No valid batch lines provided.")
            if parse_bool(join_outputs) and output_format.lower() != "wav":
                raise SystemExit("Join outputs requires WAV to avoid quality loss. Set output_format=wav.")

            update(task_id, message="building voice clone prompt")
            prompt_items = model_obj.create_voice_clone_prompt(
                ref_audio=str(ref_path),
                ref_text=ref_text_local,
                x_vector_only_mode=parse_bool(x_vector_only),
            )

            join_segments: List[np.ndarray] = []
            join_sr: Optional[int] = None

            index = 1
            for batch in chunked(items, chunk_size):
                update(task_id, message=f"generating batch {index}-{index + len(batch) - 1}")
                langs = [language] * len(batch)
                wavs, sr = model_obj.generate_voice_clone(
                    text=batch,
                    language=langs,
                    voice_clone_prompt=prompt_items,
                    **extra,
                )
                for wav in wavs:
                    out_name = f"{out_prefix}_{index:04d}.{output_format}"
                    out_path = output_dir / out_name
                    write_audio(out_path, wav, sr)
                    outputs.append({"file": f"/outputs/{task_id}/{out_name}"})
                    if parse_bool(join_outputs):
                        join_segments.append(np.asarray(wav))
                        join_sr = sr if join_sr is None else join_sr
                        if join_sr != sr:
                            raise SystemExit("Sample rate mismatch across batch outputs.")
                    index += 1

            if parse_bool(join_outputs) and join_segments:
                update(task_id, message="joining outputs")
                joined_audio = np.concatenate([seg.reshape(-1) for seg in join_segments])
                joined_name = f"{out_prefix}_joined.wav"
                joined_path = output_dir / joined_name
                write_audio(joined_path, joined_audio, join_sr or sr)
                outputs.append({"file": f"/outputs/{task_id}/{joined_name}"})
        else:
            if not text:
                raise SystemExit("Single mode requires text.")
            update(task_id, message="generating")
            wavs, sr = model_obj.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=str(ref_path),
                ref_text=ref_text_local,
                x_vector_only_mode=parse_bool(x_vector_only),
                **extra,
            )
            out_name = f"clone.{output_format}"
            out_path = output_dir / out_name
            write_audio(out_path, wavs[0], sr)
            outputs.append({"file": f"/outputs/{task_id}/{out_name}"})

        if parse_bool(apply_fx):
            update(task_id, message="applying fx")
            fx_data = {}
            if fx_config:
                try:
                    fx_data = json.loads(fx_config)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"Invalid fx_config JSON: {exc}") from exc
                if not isinstance(fx_data, dict):
                    raise SystemExit("fx_config must be a JSON object")
            fx_cfg = config_from_dict(fx_data)
            fx_outputs: List[Dict[str, Any]] = []
            for item in outputs:
                rel_path = item["file"].replace("/outputs/", "")
                in_path = OUTPUT_DIR / rel_path
                out_path = in_path.with_name(f"{in_path.stem}_fx{in_path.suffix}")
                result = apply_fx_file(in_path, out_path, fx_cfg)
                fx_outputs.append({"file": f"/outputs/{task_id}/{result.name}"})
            outputs.extend(fx_outputs)

        return outputs

    task = TASKS.create("clone", params, run)
    return JSONResponse(task)


@app.post("/api/fx")
async def apply_fx(
    input_audio: UploadFile = File(...),
    fx_config: str | None = Form(None),
):
    ensure_dirs()
    suffix = safe_suffix(input_audio.filename or "audio.wav")
    task_dir = UPLOAD_DIR / uuid.uuid4().hex[:10]
    task_dir.mkdir(parents=True, exist_ok=True)
    in_path = task_dir / f"fx_input{suffix}"
    with in_path.open("wb") as handle:
        handle.write(await input_audio.read())

    params = {"apply_fx": True}

    def run(task_id: str, update):
        update(task_id, message="applying fx")
        fx_data = {}
        if fx_config:
            try:
                fx_data = json.loads(fx_config)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid fx_config JSON: {exc}") from exc
            if not isinstance(fx_data, dict):
                raise SystemExit("fx_config must be a JSON object")
        fx_cfg = config_from_dict(fx_data)

        output_dir = OUTPUT_DIR / task_id
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"fx_output{suffix}"
        result = apply_fx_file(in_path, out_path, fx_cfg)
        return [{"file": f"/outputs/{task_id}/{result.name}"}]

    task = TASKS.create("fx", params, run)
    return JSONResponse(task)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
