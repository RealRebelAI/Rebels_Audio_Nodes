"""
Rebel Audio Nodes - Audio manipulation for ComfyUI LTX workflows
RealRebelAI

One node does everything:
    RebelAudioEdit — trim, pitch, tempo, filter, EQ, compress, reverb, chorus, gain, fade
                     with built-in preview player and pre-LTX memory flush

Section pair (for region-only edits):
    RebelAudioSectionExtract  — pull a time region out
    RebelAudioSectionMerge    — stitch it back in with crossfade

Requires:
    pip install pedalboard librosa soundfile   (auto-installed via __init__.py)
"""

import gc
import os
import uuid
import ctypes

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------
def _require_pedalboard():
    try:
        import pedalboard
        return pedalboard
    except ImportError:
        raise ImportError(
            "[RebelAudio] pedalboard not found. Run:\n"
            "  python_embeded\\python.exe -m pip install pedalboard"
        )

def _require_librosa():
    try:
        import librosa
        return librosa
    except ImportError:
        raise ImportError(
            "[RebelAudio] librosa not found. Run:\n"
            "  python_embeded\\python.exe -m pip install librosa"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_numpy(audio: dict) -> tuple[np.ndarray, int]:
    """ComfyUI AUDIO dict -> float32 numpy [channels, samples]"""
    arr = audio["waveform"][0].cpu().numpy().astype(np.float32)
    return arr, audio["sample_rate"]


def _to_audio(arr: np.ndarray, sr: int) -> dict:
    """float32 numpy [channels, samples] -> ComfyUI AUDIO dict"""
    return {"waveform": torch.from_numpy(arr).unsqueeze(0), "sample_rate": sr}


def _apply_pedalboard(arr: np.ndarray, sr: int, plugins: list) -> np.ndarray:
    pb = _require_pedalboard()
    board = pb.Pedalboard(plugins)
    return np.stack([board(ch, sr, reset=True) for ch in arr], axis=0)


def _flush_memory():
    """Full memory flush: GC, VRAM, librosa cache, Windows working set trim."""
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        import librosa
        if hasattr(librosa.filters, 'cache'):
            librosa.filters.cache.clear()
    except Exception:
        pass
    try:
        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Node: RebelAudioEdit  (the one node)
# ---------------------------------------------------------------------------
class RebelAudioEdit:
    """
    All-in-one audio editor for ComfyUI. Enable only what you need.

    Processing order (fixed internal chain):
        Trim -> Pitch Shift -> Time Stretch -> Filter -> EQ ->
        Compressor -> Reverb -> Chorus -> Gain -> Fade

    Built-in:
        preview  -- renders a playback widget on the node after processing
        flush    -- wipes RAM/VRAM before handing off to LTX
    """

    CATEGORY     = "RebelAudio"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION     = "process"
    OUTPUT_NODE  = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),

                # Trim
                "trim":           ("BOOLEAN", {"default": False}),
                "trim_start_sec": ("FLOAT",   {"default": 0.0,  "min": 0.0,    "max": 3600.0, "step": 0.05}),
                "trim_end_sec":   ("FLOAT",   {"default": 0.0,  "min": 0.0,    "max": 3600.0, "step": 0.05,
                                               "tooltip": "0 = keep to end of file"}),

                # Pitch Shift
                "pitch":           ("BOOLEAN", {"default": False}),
                "pitch_semitones": ("FLOAT",   {"default": 0.0, "min": -24.0, "max": 24.0, "step": 0.5,
                                                "tooltip": "+12 = octave up, -12 = octave down"}),

                # Time Stretch
                "tempo":                ("BOOLEAN", {"default": False}),
                "tempo_rate":           ("FLOAT",   {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05,
                                                     "tooltip": "1.0 = original. 2.0 = double speed."}),
                "tempo_preserve_pitch": ("BOOLEAN", {"default": True,
                                                     "tooltip": "True = phase vocoder (tempo only). False = resample (pitch follows speed)."}),

                # Filter
                "filter":           ("BOOLEAN",                          {"default": False}),
                "filter_type":      (["lowpass", "highpass", "bandpass"], {"default": "lowpass"}),
                "filter_cutoff_hz": ("FLOAT",                            {"default": 8000.0, "min": 20.0, "max": 20000.0, "step": 10.0}),
                "filter_q":         ("FLOAT",                            {"default": 0.707,  "min": 0.1,  "max": 10.0,   "step": 0.1}),

                # EQ
                "eq":          ("BOOLEAN", {"default": False}),
                "eq_low_hz":   ("FLOAT",   {"default": 200.0,  "min": 20.0,   "max": 2000.0,  "step": 10.0}),
                "eq_low_db":   ("FLOAT",   {"default": 0.0,    "min": -18.0,  "max": 18.0,    "step": 0.5}),
                "eq_mid_hz":   ("FLOAT",   {"default": 1000.0, "min": 100.0,  "max": 8000.0,  "step": 10.0}),
                "eq_mid_db":   ("FLOAT",   {"default": 0.0,    "min": -18.0,  "max": 18.0,    "step": 0.5}),
                "eq_mid_q":    ("FLOAT",   {"default": 1.0,    "min": 0.1,    "max": 10.0,    "step": 0.1}),
                "eq_high_hz":  ("FLOAT",   {"default": 6000.0, "min": 2000.0, "max": 20000.0, "step": 100.0}),
                "eq_high_db":  ("FLOAT",   {"default": 0.0,    "min": -18.0,  "max": 18.0,    "step": 0.5}),

                # Compressor
                "compress":             ("BOOLEAN", {"default": False}),
                "compress_threshold":   ("FLOAT",   {"default": -18.0, "min": -60.0, "max": 0.0,    "step": 1.0}),
                "compress_ratio":       ("FLOAT",   {"default": 4.0,   "min": 1.0,   "max": 20.0,   "step": 0.5}),
                "compress_attack_ms":   ("FLOAT",   {"default": 10.0,  "min": 0.1,   "max": 500.0,  "step": 1.0}),
                "compress_release_ms":  ("FLOAT",   {"default": 100.0, "min": 1.0,   "max": 2000.0, "step": 10.0}),
                "compress_makeup_db":   ("FLOAT",   {"default": 0.0,   "min": 0.0,   "max": 24.0,   "step": 0.5}),

                # Reverb
                "reverb":          ("BOOLEAN", {"default": False}),
                "reverb_room":     ("FLOAT",   {"default": 0.3,  "min": 0.0, "max": 1.0, "step": 0.05}),
                "reverb_damping":  ("FLOAT",   {"default": 0.5,  "min": 0.0, "max": 1.0, "step": 0.05}),
                "reverb_wet":      ("FLOAT",   {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),
                "reverb_dry":      ("FLOAT",   {"default": 0.8,  "min": 0.0, "max": 1.0, "step": 0.05}),
                "reverb_width":    ("FLOAT",   {"default": 1.0,  "min": 0.0, "max": 1.0, "step": 0.1}),

                # Chorus
                "chorus":           ("BOOLEAN", {"default": False}),
                "chorus_rate_hz":   ("FLOAT",   {"default": 1.5, "min": 0.1, "max": 10.0, "step": 0.1}),
                "chorus_depth":     ("FLOAT",   {"default": 0.1, "min": 0.0, "max": 1.0,  "step": 0.05}),
                "chorus_delay_ms":  ("FLOAT",   {"default": 7.0, "min": 0.0, "max": 50.0, "step": 0.5}),
                "chorus_mix":       ("FLOAT",   {"default": 0.5, "min": 0.0, "max": 1.0,  "step": 0.05}),

                # Gain / Normalize
                "gain_db":   ("FLOAT",   {"default": 0.0, "min": -40.0, "max": 20.0, "step": 0.5}),
                "normalize": ("BOOLEAN", {"default": False,
                                          "tooltip": "Normalize to 0dBFS peak before applying gain."}),

                # Fade
                "fade_in_sec":  ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "fade_out_sec": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1}),

                # Output options
                "preview": ("BOOLEAN", {"default": True,
                                        "tooltip": "Render an in-node playback widget after processing."}),
                "flush":   ("BOOLEAN", {"default": True,
                                        "tooltip": "Flush RAM/VRAM after processing. Enable when this is the last audio node before LTX."}),
            }
        }

    def process(
        self, audio,
        trim, trim_start_sec, trim_end_sec,
        pitch, pitch_semitones,
        tempo, tempo_rate, tempo_preserve_pitch,
        filter, filter_type, filter_cutoff_hz, filter_q,
        eq, eq_low_hz, eq_low_db, eq_mid_hz, eq_mid_db, eq_mid_q, eq_high_hz, eq_high_db,
        compress, compress_threshold, compress_ratio, compress_attack_ms, compress_release_ms, compress_makeup_db,
        reverb, reverb_room, reverb_damping, reverb_wet, reverb_dry, reverb_width,
        chorus, chorus_rate_hz, chorus_depth, chorus_delay_ms, chorus_mix,
        gain_db, normalize,
        fade_in_sec, fade_out_sec,
        preview, flush,
    ):
        arr, sr  = _to_numpy(audio)
        scratch  = []   # intermediates to clean up

        # 1. Trim
        if trim:
            total = arr.shape[-1]
            s = int(trim_start_sec * sr)
            e = int(trim_end_sec * sr) if trim_end_sec > 0 else total
            e = min(e, total)
            scratch.append(arr); arr = arr[:, s:e].copy()

        # 2. Pitch Shift
        if pitch and pitch_semitones != 0.0:
            pb = _require_pedalboard()
            scratch.append(arr)
            arr = _apply_pedalboard(arr, sr, [pb.PitchShift(semitones=pitch_semitones)])

        # 3. Time Stretch
        if tempo and tempo_rate != 1.0:
            if tempo_preserve_pitch:
                librosa = _require_librosa()
                stretched = np.stack(
                    [librosa.effects.time_stretch(ch, rate=tempo_rate) for ch in arr], axis=0
                )
            else:
                import scipy.signal as sig
                new_len   = int(arr.shape[-1] / tempo_rate)
                stretched = np.stack(
                    [sig.resample(ch, new_len).astype(np.float32) for ch in arr], axis=0
                )
            scratch.append(arr); arr = stretched

        # 4. Filter
        if filter:
            import scipy.signal as sig
            nyq = sr / 2.0
            c   = min(filter_cutoff_hz / nyq, 0.99)
            if filter_type == "lowpass":
                b, a = sig.butter(4, c, btype="low")
            elif filter_type == "highpass":
                b, a = sig.butter(4, c, btype="high")
            else:
                bw   = c / max(filter_q, 0.1)
                b, a = sig.butter(4, [max(c - bw/2, 0.001), min(c + bw/2, 0.999)], btype="band")
            scratch.append(arr)
            arr = np.stack([sig.filtfilt(b, a, ch).astype(np.float32) for ch in arr], axis=0)

        # 5. EQ
        if eq:
            pb      = _require_pedalboard()
            plugins = []
            if eq_low_db  != 0.0: plugins.append(pb.LowShelfFilter( cutoff_frequency_hz=eq_low_hz,  gain_db=eq_low_db))
            if eq_mid_db  != 0.0: plugins.append(pb.PeakFilter(     cutoff_frequency_hz=eq_mid_hz,  gain_db=eq_mid_db, q=eq_mid_q))
            if eq_high_db != 0.0: plugins.append(pb.HighShelfFilter(cutoff_frequency_hz=eq_high_hz, gain_db=eq_high_db))
            if plugins:
                scratch.append(arr)
                arr = _apply_pedalboard(arr, sr, plugins)

        # 6. Compressor
        if compress:
            pb = _require_pedalboard()
            scratch.append(arr)
            arr = _apply_pedalboard(arr, sr, [
                pb.Compressor(
                    threshold_db=compress_threshold,
                    ratio=compress_ratio,
                    attack_ms=compress_attack_ms,
                    release_ms=compress_release_ms,
                ),
                pb.Gain(gain_db=compress_makeup_db),
            ])

        # 7. Reverb
        if reverb:
            pb = _require_pedalboard()
            scratch.append(arr)
            arr = _apply_pedalboard(arr, sr, [
                pb.Reverb(
                    room_size=reverb_room,
                    damping=reverb_damping,
                    wet_level=reverb_wet,
                    dry_level=reverb_dry,
                    width=reverb_width,
                )
            ])

        # 8. Chorus
        if chorus:
            pb = _require_pedalboard()
            scratch.append(arr)
            arr = _apply_pedalboard(arr, sr, [
                pb.Chorus(
                    rate_hz=chorus_rate_hz,
                    depth=chorus_depth,
                    centre_delay_ms=chorus_delay_ms,
                    mix=chorus_mix,
                )
            ])

        # 9. Gain / Normalize
        if normalize:
            peak = np.abs(arr).max()
            if peak > 0:
                scratch.append(arr); arr = (arr / peak).astype(np.float32)

        if gain_db != 0.0:
            linear = 10 ** (gain_db / 20.0)
            scratch.append(arr)
            arr = np.clip(arr * linear, -1.0, 1.0).astype(np.float32)

        # 10. Fade
        if fade_in_sec > 0 or fade_out_sec > 0:
            n        = arr.shape[-1]
            envelope = np.ones(n, dtype=np.float32)
            fi = int(fade_in_sec  * sr)
            fo = int(fade_out_sec * sr)
            if fi > 0: envelope[:fi]  *= np.linspace(0.0, 1.0, fi)
            if fo > 0: envelope[-fo:] *= np.linspace(1.0, 0.0, fo)
            scratch.append(arr)
            arr = (arr * envelope).astype(np.float32)

        # Build result
        result_audio = _to_audio(arr, sr)

        # Clean up all intermediate arrays
        for a in scratch:
            try: del a
            except Exception: pass
        del scratch, arr
        gc.collect()

        # Preview
        ui_payload = {}
        if preview:
            try:
                import folder_paths
                import soundfile as sf
                p_arr, p_sr = _to_numpy(result_audio)
                filename    = f"rebel_audio_{uuid.uuid4().hex[:8]}.wav"
                sf.write(os.path.join(folder_paths.get_temp_directory(), filename), p_arr.T, p_sr, subtype="PCM_16")
                del p_arr
                ui_payload = {"audio": [{"filename": filename, "subfolder": "", "type": "temp"}]}
            except Exception as e:
                print(f"[RebelAudioEdit] Preview error: {e}")

        # Flush
        if flush:
            print("[RebelAudioEdit] Flushing memory before LTX...")
            _flush_memory()
            print("[RebelAudioEdit] Flush complete. Ready for LTX.")

        if ui_payload:
            return {"ui": ui_payload, "result": (result_audio,)}
        return (result_audio,)


# ---------------------------------------------------------------------------
# Node: Section Extract
# ---------------------------------------------------------------------------
class RebelAudioSectionExtract:
    """
    Pull a time region out so you can run RebelAudioEdit on just that section,
    then stitch it back with RebelAudioSectionMerge.

    Wire:
        Load Audio -> Section Extract -> RebelAudioEdit -> Section Merge -> LTX
                              |__ original __________________________________|
    """

    CATEGORY     = "RebelAudio/Section"
    RETURN_TYPES = ("AUDIO", "AUDIO", "FLOAT")
    RETURN_NAMES = ("section", "original", "start_sec")
    FUNCTION     = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio":     ("AUDIO",),
                "start_sec": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 3600.0, "step": 0.05}),
                "end_sec":   ("FLOAT", {"default": 2.0, "min": 0.0, "max": 3600.0, "step": 0.05,
                                         "tooltip": "0 = end of file"}),
            }
        }

    def process(self, audio, start_sec, end_sec):
        arr, sr = _to_numpy(audio)
        total   = arr.shape[-1] / sr
        end_sec = total if end_sec <= 0 or end_sec > total else end_sec

        if start_sec >= end_sec:
            raise ValueError("[RebelAudioSectionExtract] start_sec must be less than end_sec")

        s = int(start_sec * sr)
        e = int(end_sec   * sr)
        section = arr[:, s:e].copy()
        del arr

        print(f"[RebelAudioSectionExtract] {start_sec:.3f}s -> {end_sec:.3f}s ({end_sec - start_sec:.3f}s)")
        return (_to_audio(section, sr), audio, float(start_sec))


# ---------------------------------------------------------------------------
# Node: Section Merge
# ---------------------------------------------------------------------------
class RebelAudioSectionMerge:
    """
    Stitch a processed section back into the original at the position it
    was extracted from. Handles duration changes. Crossfade prevents clicks.
    """

    CATEGORY     = "RebelAudio/Section"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION     = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original":     ("AUDIO",),
                "section":      ("AUDIO",),
                "start_sec":    ("FLOAT", {"default": 0.1,  "min": 0.0, "max": 3600.0, "step": 0.05,
                                            "tooltip": "Wire directly from Section Extract output."}),
                "crossfade_ms": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 200.0,  "step": 5.0,
                                            "tooltip": "0 = hard cut. 10-30ms is transparent."}),
            }
        }

    def process(self, original, section, start_sec, crossfade_ms):
        import scipy.signal as sig

        orig, sr   = _to_numpy(original)
        sect, s_sr = _to_numpy(section)

        if s_sr != sr:
            new_len = int(sect.shape[-1] * sr / s_sr)
            sect    = np.stack([sig.resample(ch, new_len).astype(np.float32) for ch in sect], axis=0)

        total          = orig.shape[-1]
        s_start        = min(int(start_sec * sr), total)
        s_len          = sect.shape[-1]
        cf             = int((crossfade_ms / 1000.0) * sr)
        orig_after_s   = min(s_start + s_len, total)

        out = np.concatenate([orig[:, :s_start], sect, orig[:, orig_after_s:]], axis=-1).astype(np.float32)

        # Crossfade at start boundary
        if cf > 0 and s_start > 0:
            cf_len = min(cf, s_start, s_len)
            fo = np.linspace(1.0, 0.0, cf_len, dtype=np.float32)
            fi = np.linspace(0.0, 1.0, cf_len, dtype=np.float32)
            out[:, s_start - cf_len : s_start]       *= fo
            out[:, s_start          : s_start + cf_len] *= fi
            out[:, s_start - cf_len : s_start]       += orig[:, s_start - cf_len : s_start] * fi

        # Crossfade at end boundary
        s_end = s_start + s_len
        if cf > 0 and orig_after_s < total:
            cf_len = min(cf, s_len, total - orig_after_s)
            if s_end + cf_len <= out.shape[-1]:
                fo = np.linspace(1.0, 0.0, cf_len, dtype=np.float32)
                fi = np.linspace(0.0, 1.0, cf_len, dtype=np.float32)
                out[:, s_end - cf_len : s_end]       *= fo
                out[:, s_end          : s_end + cf_len] *= fi
                out[:, s_end - cf_len : s_end]       += orig[:, orig_after_s - cf_len : orig_after_s] * fi

        out    = np.clip(out, -1.0, 1.0).astype(np.float32)
        result = _to_audio(out, sr)
        del orig, sect, out
        gc.collect()

        print(f"[RebelAudioSectionMerge] Merged at {start_sec:.3f}s, total output {result['waveform'].shape[-1] / sr:.3f}s")
        return (result,)


# ---------------------------------------------------------------------------
# NODE MAPPINGS
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "RebelAudioEdit":           RebelAudioEdit,
    "RebelAudioSectionExtract": RebelAudioSectionExtract,
    "RebelAudioSectionMerge":   RebelAudioSectionMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RebelAudioEdit":           "🎛️ Rebel Audio Edit",
    "RebelAudioSectionExtract": "✂️ Rebel Audio - Section Extract",
    "RebelAudioSectionMerge":   "🔗 Rebel Audio - Section Merge",
}