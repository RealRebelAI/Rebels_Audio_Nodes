# 🎛️ Rebels Audio Nodes — ComfyUI

**Audio editing nodes for ComfyUI, built for LTX Video audio-to-video workflows.**

Part of the [Rebel AI](https://www.youtube.com/@RealRebelAI) ecosystem — making frontier AI accessible on consumer hardware.

---

## What This Is

Edit audio inside your ComfyUI workflow before it hits video generation. Export from FL Studio or any DAW, load it, shape it, feed it into LTX Video. No external audio editor needed — the whole chain stays in ComfyUI.

---

## Nodes

### 🎛️ Rebel Audio Edit
The main node. Everything in one place — enable only what you need.

| Section | Controls |
|---|---|
| **Trim** | Start / end time in seconds |
| **Pitch Shift** | ±24 semitones, duration unchanged |
| **Time Stretch** | Speed multiplier with pitch-lock (phase vocoder) or pitch-follows-speed (resample) |
| **Filter** | Lowpass / Highpass / Bandpass — cutoff Hz + Q |
| **EQ** | 3-band: low shelf, mid peak, high shelf — all in dB |
| **Compressor** | Threshold, ratio, attack/release, makeup gain |
| **Reverb** | Room size, damping, wet/dry, stereo width |
| **Chorus** | Rate, depth, delay, mix |
| **Gain** | dB adjustment + optional normalize to 0dBFS |
| **Fade** | Fade in / fade out in seconds |
| **Preview** | Toggle — renders a playback widget directly on the node |
| **Flush** | Toggle — wipes RAM and VRAM before handing off to LTX |

Processing always runs in this fixed internal order:
```
Trim → Pitch → Tempo → Filter → EQ → Compressor → Reverb → Chorus → Gain → Fade
```

**Standard workflow (full clip):**
```
Load Audio → 🎛️ Rebel Audio Edit → LTX Video
```

---

### ✂️ Rebel Audio - Section Extract + 🔗 Rebel Audio - Section Merge

For when you want to apply effects to a specific time region only, leaving the rest of the clip untouched.

**Section workflow:**
```
Load Audio
    │
    ▼
Section Extract (start_sec=1.0, end_sec=3.0)
    ├─ section   ──────────────────► Rebel Audio Edit ──► Section Merge (section)
    ├─ original  ────────────────────────────────────────► Section Merge (original)
    ├─ start_sec ────────────────────────────────────────► Section Merge (start_sec)
    └─ end_sec   ────────────────────────────────────────► Section Merge (end_sec)
                                                                  │
                                                                  ▼
                                                            LTX Video
```

**Important:** wire `start_sec` AND `end_sec` from Section Extract directly into Section Merge. Section Merge uses `end_sec` to know exactly where to resume the original audio — if you don't wire it, the splice point will be wrong.

Section Merge also has a `crossfade_ms` parameter (default 20ms) that smooths the join at both cut points to prevent clicks. Set to 0 for a hard cut.

---

## Installation

### Option 1 — ComfyUI Manager
Search **Rebels Audio Nodes** and install directly.

### Option 2 — Manual
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/RealRebelAI/Rebels_Audio_Nodes
```
Restart ComfyUI. Dependencies install automatically on first load.

### Option 3 — Manual dep install (if auto-install fails)
```bash
# Windows portable
python_embeded\python.exe -m pip install pedalboard librosa soundfile

# Standard venv
pip install pedalboard librosa soundfile
```

---

## Dependencies

| Library | Used For |
|---|---|
| [pedalboard](https://github.com/spotify/pedalboard) | Pitch shift, reverb, EQ, filter, compressor, chorus (Spotify / JUCE-backed) |
| [librosa](https://librosa.org/) | Phase vocoder time stretch (tempo without pitch change) |
| [soundfile](https://python-soundfile.readthedocs.io/) | Writing preview audio to temp files |

All three install automatically on first ComfyUI load.

---

## Memory

All processing is CPU-side — no VRAM is used during audio editing. With the **Flush** toggle enabled on Rebel Audio Edit, all intermediate buffers, numpy arrays, and librosa cache are cleared before LTX begins. Your GPU is untouched until LTX starts.

Tested on RTX 3070 8GB / 16GB RAM.

---

## Part of the Rebel AI Ecosystem

- 🎬 [YouTube](https://www.youtube.com/@RealRebelAI) — tutorials, workflows, model releases
- 🤗 [HuggingFace](https://huggingface.co/RealRebelAI) — quantized models for consumer hardware

---

## License
MIT
