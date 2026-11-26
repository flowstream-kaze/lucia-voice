# Lucia's voice — a Kokoro-82M blend, and how to rebuild it

**`bf_emma50_lily50`** — Lucia's **Kokoro** voice: an equal-weight
interpolation of two official Kokoro voice banks (`bf_emma` + `bf_lily`), British English,
speed 1.0, running on **Kokoro-82M v1.0** on CPU. Nothing is trained, fine-tuned or cloned:
the voice is two published Apache-2.0 voice banks and one average.

This repo is the reference for **what the voice is** and **how to stand it up on another
machine**. The HTTP server that serves it on the LAN lives in the Pi repository at
`integrations/voice/tts/` (`tts_server.py`); everything here is the voice identity, the
pinned weights, and the steps.

## Engines and names — read this first if you came from the app

One shared speech service fronts several engines, and each keeps its own default in
`C:\AI\VoiceChat\voices\settings.json`. Switching engine takes 15-40 s from the Voices page.

| engine | its default voice |
| --- | --- |
| `clone` — Qwen3-TTS 1.7B Base (cloned or designed voices) | `warm_british_1` |
| `clone-fast` — Qwen3-TTS 12Hz 0.6B Base | `warm_british_1` |
| `preset` — Qwen3-TTS 1.7B CustomVoice (nine studio speakers, style control) | `serena` |
| `kokoro` — **Kokoro-82M v1.0** | **`bf_emma50_lily50`** — this repo |
| `breeze` — Breeze-TTS-2 3B, directorable (non-commercial licence) | `b-warm_british_1` |

**`warm_british_1` is not a Kokoro voice** and cannot be handed to Kokoro. It is an original
voice designed with Qwen3-TTS VoiceDesign — its folder holds a `voice.json` with the design
prompt, a description and a seed (20261010) — and the Kokoro bank has no such name: all 54
entries follow `<language><gender>_<name>` (`af_`, `bf_`, `bm_`, …), so any other name is an
app-side label. As of this writing it is also what the assistant actually speaks: `/health`
on the shared service reports `qwen3-tts-clone-cuda` / `warm_british_1`, while the Kokoro
table still describes `bf_emma50_lily50` as "Pi's original voice" — which is exactly what it
is: where Lucia's voice started, on the engine that is no longer selected by default. This
repo documents that Kokoro side, the part reproducible from public weights — not necessarily
the voice you hear today.

## How the blend was achieved

Kokoro is StyleTTS-2 shaped: a voice is not one embedding but a **style bank** — one
256-dimensional style vector per phoneme token, `[510, 1, 256]` in v1.0. At synthesis time
the model picks the row for each token in the sentence. So blending means averaging the
**whole bank**, element by element, and handing the result back:

```python
style = 0.5 * bank('bf_emma') + 0.5 * bank('bf_lily')   # shape stays [510, 1, 256]
```

That is the entire trick, and [`voice_presets.py`](voice_presets.py) is the entire
implementation (a dozen lines). Averaging a single vector instead of the bank is the usual
mistake — it survives a smoke test and then flattens the prosody, because the per-token
lookup has nothing left to select from.

Facts worth having:

- The banks are `[510, 1, 256]` float32. Their SHA-256 over the raw array bytes:

  | voice | sha256 |
  | --- | --- |
  | `bf_emma` | `669fe0647f9dd04fcab92f1439a40eeb4c8b4ab1f82e4996fe3d918ce4a63b73` |
  | `bf_lily` | `5e0ee32ebe64a467124976b14e69590746f1c4ce41a12b587a50c862edfea335` |
  | **the 50/50 blend** | `188cc063f7094421bfa0f8eedb3d64ce8c6791c7da6bbb7f27928b74b87fa4aa` |

  Those two parents average to the third, exactly, and the blend hash has been reproduced on
  three different machines from the pinned `voices-v1.0.bin` — the production Windows host
  (CPU and CUDA provider) and a Debian x86-64 container. **Audio** is not byte-identical
  across providers or even between two CPU runs of the same text; only the style bank is
  deterministic. Verify a new host by ear, and verify the blend by hash.
- The blend is checkpoint-specific. Style banks live in that model's style space, and
  `bf_emma`/`bf_lily` exist only in Kokoro **v1.0** — v0.19 shipped one language and ten
  American voices and has no British voices to average. Pin v1.0.
- The official PyTorch Kokoro package does the same averaging internally, so
  `pipeline(text, voice='bf_emma,bf_lily')` on `KPipeline(lang_code='b')` is this voice
  without any code from this repo.
- It is an interpolation of two voices that exist, not a fifth official preset.

## How this particular voice was picked

1. A reference clip was level-matched against the four British **female** v1.0 presets
   (`bf_emma`, `bf_lily`, `bf_isabella`, `bf_alice`) plus three Emma/Lily ratios
   (75/25, **50/50**, 25/75) — seven candidates, one standard sentence each, reference
   first, one-second gaps, in a single listening file.
2. Rough pitch and spectral distance ranked `bf_emma` and the 75/25 blend first. That
   measurement was treated as a shortlist, not a verdict: it does not see accent or
   personality.
3. The choice was made by listening. **50/50** won — Emma's clarity with Lily's lower,
   less breathy register — and was approved at `en-gb`, speed 1.0.
4. GPU vs CPU was auditioned separately (CPU clip, one second silence, GPU clip, same
   bank): indistinguishable by ear, so the runtime was chosen on cost, not on sound.

All of it is reproducible with the files here: render the candidates, pick by ear.

## What it costs

Measured on a desktop CPU with 4 workers versus a RTX 5070 Ti, same FP32 model, same bank:

| | warm real-time factor | model/session setup | process private memory |
| --- | --- | --- | --- |
| CPU (4 workers) | ~5x realtime (RTF ~0.21) | 865 ms | 1.78 GiB |
| 5070 Ti | ~14-21x realtime (RTF ~0.05-0.07) | 2803 ms | 3.63 GiB |

GPU is 3-4x faster, but a resident GPU worker held ~1.24 GiB above the card's baseline and
blocked the image service's cold-start memory gate — so **CPU is the production choice**. A
short sentence synthesizes in well under half a second there, which is why CPU was good
enough to ship.

## Pinned weights

Apache-2.0, 354 MB total, deliberately **not** in git — `download-assets.py` fetches them
and checks size plus SHA-256:

| file | bytes | sha256 |
| --- | ---: | --- |
| `kokoro-v1.0.onnx` | 325,505,369 | `beb0d1848dee9a49da392cc3df26958d46cfa35d321edf434f52949153f0df3a` |
| `voices-v1.0.bin` | 28,214,398 | `bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d` |

Source: [thewh1teagle/kokoro-onnx `model-files-v1.1`](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.1),
an ONNX export of [hexgrad/Kokoro-82M v1.0](https://huggingface.co/hexgrad/Kokoro-82M).

## Files

| file | what it is |
| --- | --- |
| `voice_presets.py` | the blend: `DEFAULT_VOICE`, `english_voices()`, `resolve_voice()` |
| `test_voice_presets.py` | tests for it, standard library only (`python3 -m unittest`) |
| `model-manifest.json` | the weight pins above |
| `download-assets.py` | fetch + verify, stdlib only |
| `requirements-cpu.txt` | the verified CPU venv pin (Python 3.12+) |
| `requirements-cpu-py311.txt` | the same, for Python 3.11 (newer numpy needs 3.12) |
| `SETUP.md` | bring-up on a new machine, and how to teach Pi about it |

## Make your own

Edit `BLEND_SOURCES` and the two weights in `voice_presets.py`, keep the whole bank, and
stay inside one language — averaging an American bank with a British one gives you a bank
that neither G2P path reads correctly. Ratios outside 25/75-75/25 tend to just sound like
the louder voice. Then listen; the pitch measurements here were only ever a shortlist.

Code in this repo is MIT. The weights are Apache-2.0 upstream and are not redistributed
here.
