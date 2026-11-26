# Setup on a new machine

Two paths. **Hear the voice** takes about three minutes and needs no server; run it first,
because it proves the weights, the pins and the blend on the new hardware. **Serve the voice**
is the LAN backend Pi actually talks to.

Nothing here is Windows-specific — `tts_server.py` is stdlib HTTP plus NumPy plus
kokoro-onnx and runs unchanged on Linux.

## 1. Hear the voice

```bash
mkdir -p ~/lucia-voice && cd ~/lucia-voice
# stage from this repo: voice_presets.py  model-manifest.json  download-assets.py
#                        requirements-cpu.txt  (or requirements-cpu-py311.txt on Python 3.11)
python3 -m venv venv && ./venv/bin/pip install -r requirements-cpu.txt
./venv/bin/python download-assets.py        # ~354 MB, verifies size + SHA-256
```

Use `requirements-cpu-py311.txt` on Python 3.11 — the two files differ only in numpy, and
mixing them in one command is a resolution conflict, not a warning. The 3.12 pin set is what
production runs; the 3.11 file is verified working here. If `python3 -m venv` reports that
`ensurepip` is not available (a slimmed distro image without `python3-venv`), the venv is
still created but has no pip: bootstrap it with
`curl -o get-pip.py https://bootstrap.pypa.io/get-pip.py && ./venv/bin/python get-pip.py`.

Then synthesize one line, in this folder:

```bash
./venv/bin/python - <<'PY'
import onnxruntime as ort, wave
import numpy as np
from kokoro_onnx import Kokoro
from voice_presets import resolve_voice

session = ort.InferenceSession('kokoro-v1.0.onnx', providers=['CPUExecutionProvider'])
engine = Kokoro.from_session(session, 'voices-v1.0.bin')
audio, rate = engine.create('Hello. This is the blend, on a new machine.',
                           voice=resolve_voice(engine, 'bf_emma50_lily50'),
                           lang='en-gb', speed=1.0)
pcm = np.rint(np.clip(np.asarray(audio), -1, 1) * 32767).astype('<i2').tobytes()
with wave.open('out.wav', 'wb') as wav:
    wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(rate); wav.writeframes(pcm)
print('wrote out.wav', len(pcm) // 2 / rate, 'seconds at', rate, 'Hz')
PY
```

Expect roughly 5x realtime on four modern CPU cores: a sentence lands in well under a
second after the model is loaded (~0.9 s setup). Compare the first syllables and the
sentence-final pitch against a known clip. **Do not compare WAV hashes** — Kokoro output is
not reproducible byte-for-byte even between two CPU runs of the same text.

Sanity checks that catch the real failure modes:

- `resolve_voice` raising `Unknown English voice` → the voice bank is not v1.0, or the model
  and bank come from different releases. Re-run `download-assets.py`.
- A flat, monotone result → something averaged a single style vector instead of the bank.
- American-sounding `bath`/`not` → the language is `en-us`; it must be `en-gb`.

## 2. Serve the voice

The backend source is `integrations/voice/tts/tts_server.py` in the Pi repository
(`flowstream-kaze/pi-kaze`). Copy `tts_server.py` and `voice_presets.py` into one folder,
with `kokoro-v1.0.onnx` and `voices-v1.0.bin` beside them — the server loads by sibling
name — plus `voice_presets.py` from this repo (same file, keep them identical):

```bash
./venv/bin/python tts_server.py --host 0.0.0.0 --port 8880 --token <private-token>
# [tts] warmed bf_emma50_lily50 cpu-threads=4 on :8880
curl -H "Authorization: Bearer <private-token>" http://<host>:8880/health
# {"ok":true,"backend":"kokoro-cpu","voice":"bf_emma50_lily50","warm":true}
```

`/health` and `/synthesize` both require the Bearer token. `--host 0.0.0.0` is a
private-LAN listener: keep it on a private network behind a host-scoped firewall rule,
never exposed. `--threads` accepts 4 or 8.

## 3. Teach Pi about it

Environment variables, or the private file `PI_HOME/agent/voice-tts.json` (mode 0600, read
at startup; environment takes precedence):

```json
{ "url": "http://<host>:8880", "token": "<private-token>" }
```

`PI_VOICE_TTS_URL` / `PI_VOICE_TTS_TOKEN`. Missing or malformed values disable synthesis
rather than failing, so a silent assistant usually means this step did not travel to the new
host. The token is never committed and never logged.

## 4. Do not copy these across

- **Threads and CPU affinity** were tuned per machine. The P-core masks in the old benchmark
  (`0x55`, `0x5555`) are only meaningful for that host's core layout. Start at 4 threads and
  leave affinity alone.
- **The scheduled task.** On Windows, `\PiVoiceTts` (boot trigger, SYSTEM) must be
  registered by hand; `Manage-Tts-Task.ps1` in the Pi repository only adjusts four settings
  of an existing task and refuses anything else. On Linux, use your own service unit.
- **Process priorities** (BelowNormal) and any GPU or game-mode ownership rules belonged to
  the old host's coexistence arrangement, not to the voice.

## Out of scope here

Speech-to-text is a separate backend (Whisper/Parakeet) in the Pi repository, and the Pi
server container deploys from its own scripts. If you want a different voice, see
"Make your own" in the README — it is the same two lines of arithmetic.
