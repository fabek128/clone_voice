# clon_voice (Qwen3-TTS)

Proyecto minimo para clonar voces con Qwen3-TTS. Incluye script listo para ejecutar y estructura de carpetas.

## Requisitos
- Python 3.10+ (recomendado 3.12)
- GPU CUDA para rendimiento aceptable (CPU funciona pero es muy lento)
- Acceso a internet para descargar modelos la primera vez

## Instalacion rapida (Windows / PowerShell)
1. python -m venv .venv
2. .\.venv\Scripts\Activate.ps1
3. Instala PyTorch adecuado para tu CUDA (o CPU) desde:
   https://pytorch.org/get-started/locally/
4. pip install -r requirements.txt
5. (Opcional) FlashAttention 2 para menos VRAM y mas velocidad:
   pip install -U flash-attn --no-build-isolation
   Requiere GPU compatible y dtype float16/bfloat16.

## Configuracion
Edita `config.json` para fijar el modelo por defecto y otros parametros. La CLI siempre tiene prioridad.

Campos utiles:
- model
- device
- dtype
- attn
- language
- out
- batch_out_dir
- batch_chunk_size
- ui_presets

## Descarga previa del modelo

python scripts/download_model.py

Opcionalmente puedes fijar `--cache-dir` o `--revision`.

## UI local (CLI)

python scripts/ui_cli.py

Incluye presets de texto en `presets.json`.

## Web UI (frontend simple)

La app web permite subir audio, definir transcript, configurar parametros, correr tareas en background y ver estados en tiempo real (SSE).

### Levantar en local (backend + frontend juntos)
1. Crear y activar entorno:
   - `python -m venv .venv`
   - `.\.venv\Scripts\Activate.ps1`
2. Instalar dependencias:
   - `pip install -r requirements.txt`
3. Levantar servidor:
   - `python app/server.py`
4. Abrir en el navegador:
   - `http://127.0.0.1:8000`

El backend sirve el frontend estatico desde `web/`, y los outputs desde `/outputs/`.

Notas:
- Requiere `ffmpeg` en PATH para exportar mp3 o procesar mp3/m4a.
- Los outputs se sirven desde `/outputs/`.

## Uso basico (clonacion)
Ejemplo con audio local y transcript:

python scripts/clone_voice.py --ref-audio data/inputs/ref.wav --ref-text "Tu texto de referencia aqui" --text "Texto a sintetizar" --language Spanish --out data/outputs/clone.wav

Ejemplo sin transcript (menos calidad, usa solo embedding del speaker):

python scripts/clone_voice.py --ref-audio data/inputs/ref.wav --x-vector-only --text "Texto a sintetizar" --language Spanish --out data/outputs/clone.wav

## Batch (cache de prompt)
Crea el prompt una sola vez con `create_voice_clone_prompt` y reutiliza para varias frases.

Crea un archivo con una frase por linea, por ejemplo `data/inputs/texts.txt` (lineas vacias o con # se ignoran).

python scripts/batch_clone.py --ref-audio data/inputs/ref.wav --ref-text "Tu texto de referencia aqui" --texts-file data/inputs/texts.txt --language Spanish --out-dir data/outputs/batch --chunk-size 8

## Post-procesado (FX con Pedalboard + pydub)
Para aplicar efectos a un WAV o MP3:

python scripts/apply_fx.py --input data/outputs/clone.wav --preset reverb --normalize

Presets disponibles: `thick`, `reverb`, `echo`, `broadcast`.

Requiere `ffmpeg` en PATH para MP3/M4A. En Windows puedes instalarlo con `choco install ffmpeg` o desde https://ffmpeg.org/download.html

## Modelos recomendados
- Qwen/Qwen3-TTS-12Hz-1.7B-Base (mejor calidad, mas VRAM)
- Qwen/Qwen3-TTS-12Hz-0.6B-Base (mas liviano)

Puedes cambiar con --model o editando config.json.

## Notas importantes
- Para clonacion, el modelo base espera ref_audio y su ref_text correspondiente. Si usas --x-vector-only no necesitas transcript pero baja la calidad.
- ref_audio puede ser ruta local, URL, base64 o un tuple (numpy_array, sample_rate).
- La primera ejecucion descargara los pesos del modelo.

## Estructura
- config.json: defaults del proyecto
- presets.json: textos predefinidos para la UI
- scripts/clone_voice.py: script principal de clonacion
- scripts/batch_clone.py: batch con cache de prompt
- scripts/download_model.py: descarga previa de pesos
- scripts/tts_config.py: carga y mezcla de config
- scripts/ui_cli.py: UI local por CLI
- scripts/apply_fx.py: post-procesado con FX
- data/inputs/: coloca tus audios de referencia aqui
- data/outputs/: salida de audios generados

## Legal y consentimiento
Solo clona voces con consentimiento explicito y para usos permitidos por la ley aplicable.
