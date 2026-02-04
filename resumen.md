# Info: clonacion de voz con Qwen3-TTS

Proyecto minimo y funcional para clonacion de voz con **Qwen3-TTS** (modelo base `Qwen/Qwen3-TTS-12Hz-1.7B-Base`). La idea fue dejar todo listo para correr local, con un flujo claro de clonacion y algunas herramientas extra para productividad.

## Lo que se implemento
- **Script principal de clonacion** (`scripts/clone_voice.py`) usando `Qwen3TTSModel.generate_voice_clone`.
- **Batch con cache de prompt** (`scripts/batch_clone.py`) para reutilizar `create_voice_clone_prompt` y generar muchas frases mas rapido.
- **UI local por CLI** (`scripts/ui_cli.py`) con presets y flujo guiado.
- **Descarga previa de pesos** (`scripts/download_model.py`) para evitar la espera de la primera corrida.
- **Config centralizada** (`config.json` + `scripts/tts_config.py`) para defaults consistentes.
- **Presets** de texto (`presets.json`) y estructura de carpetas `data/inputs` / `data/outputs`.

## Librerias usadas
- `qwen-tts` como wrapper principal del modelo.
- `torch` (PyTorch) para la ejecucion del modelo.
- `soundfile` para escritura de WAV.
- `numpy` como dependencia base.
- `huggingface_hub` para descargar modelos (en `download_model.py`).
- **Opcional**: `flash-attn` para mejorar rendimiento/VRAM en GPU compatibles.

## Problemas que encontramos (y solucionamos)
- **Version de `qwen-tts`**: el requirement `>=0.1.0` no existia en PyPI, asi que lo ajustamos a `>=0.0.5` para que `pip install -r requirements.txt` funcione.
- **Clonacion sin transcript**: el modelo **requiere `ref_text`** si no se usa `--x-vector-only`. Si falta, la ejecucion falla. Se documentaron ambas opciones y el CLI guia ese flujo.

## Buenas practicas que quedaron documentadas
- Fijar `--language Spanish` para evitar deteccion automatica cuando se busca mas control.
- Usar audio de referencia con buena calidad (ideal WAV mono 16-bit, >=24 kHz).
- Evitar `--x-vector-only` si se busca mayor fidelidad.
- Usar batch con cache de prompt cuando se generan muchas frases.
