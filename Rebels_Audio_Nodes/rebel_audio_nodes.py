"""
Rebel Audio Nodes - Audio manipulation for ComfyUI LTX workflows
RealRebelAI

Nodes:
    RebelAudioEdit          — all-in-one audio editor with toggleable effect sections
    RebelAudioSectionExtract — pull a time region out for isolated editing
    RebelAudioSectionMerge   — stitch the processed region back in

Requires:  pip install pedalboard librosa soundfile  (auto-installed via __init__.py)
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
        raise ImportError("[RebelAudio] pip install pedalboard")

def _require_librosa():
    try:
        import librosa
        return librosa
    except ImportError:
        raise ImportError("[RebelAudio] pip install librosa")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_numpy(audio):
    return audio["waveform"][0].cpu().numpy().astype(np.float32), audio["sample_rate"]

def _to_audio(arr, sr):
    return {"waveform": torch.from_numpy(arr).unsqueeze(0), "sample_rate": sr}

def _apply_pedalboard(arr, sr, plugins):
    pb = _require_pedalboard()
    board = pb.Pedalboard(plugins)
    return np.stack([board(ch, sr, reset=True) for ch in arr], axis=0)

def _flush_memory():
    gc.collect(0); gc.collect(1); gc.collect(2)
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception: pass
    try:
        import librosa
        if hasattr(librosa.filters, "cache"): librosa.filters.cache.clear()
    except Exception: pass
    try:
        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
    except Exception: pass


# ---------------------------------------------------------------------------
# Node: RebelAudioEdit
# ---------------------------------------------------------------------------
class RebelAudioEdit:
    """
    All-in-one audio editor. Enable only what you need.
    Processing order: Trim > Pitch > Tempo > Filter > EQ > Compressor > Reverb > Chorus > Gain > Fade
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
                "pitch_semitones": ("FLOAT",   {"default": 0.0, "min": -24.0, "max": 24.0, "step": 0.5}),

                # Time Stretch
                "tempo":                ("BOOLEAN", {"default": False}),
                "tempo_rate":           ("FLOAT",   {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05}),
                "tempo_preserve_pitch": ("BOOLEAN", {"default": True}),

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

                # Gain
                "gain_db":   ("FLOAT",   {"default": 0.0, "min": -40.0, "max": 20.0, "step": 0.5}),
                "normalize": ("BOOLEAN", {"default": False}),

                # Fade
                "fade_in_sec":  ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "fade_out_sec": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1}),

                # Output
                "preview": ("BOOLEAN", {"default": True}),
                "flush":   ("BOOLEAN", {"default": True}),
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
        arr, sr = _to_numpy(audio)
        scratch = []

        # 1. Trim
        if trim:
            total = arr.shape[-1]
            s = int(trim_start_sec * sr)
            e = int(trim_end_sec * sr) if trim_end_sec > 0 else total
            scratch.append(arr); arr = arr[:, s:min(e, total)].copy()

        # 2. Pitch Shift
        if pitch and pitch_semitones != 0.0:
            pb = _require_pedalboard()
            scratch.append(arr)
            arr = _apply_pedalboard(arr, sr, [pb.PitchShift(semitones=pitch_semitones)])

        # 3. Time Stretch
        if tempo and tempo_rate != 1.0:
            if tempo_preserve_pitch:
                librosa = _require_librosa()
                new = np.stack([librosa.effects.time_stretch(ch, rate=tempo_rate) for ch in arr], axis=0)
            else:
                import scipy.signal as sig
                new = np.stack([sig.resample(ch, int(arr.shape[-1] / tempo_rate)).astype(np.float32) for ch in arr], axis=0)
            scratch.append(arr); arr = new

        # 4. Filter
        if filter:
            import scipy.signal as sig
            nyq = sr / 2.0; c = min(filter_cutoff_hz / nyq, 0.99)
            if filter_type == "lowpass":    b, a = sig.butter(4, c, btype="low")
            elif filter_type == "highpass": b, a = sig.butter(4, c, btype="high")
            else:
                bw = c / max(filter_q, 0.1)
                b, a = sig.butter(4, [max(c-bw/2, 0.001), min(c+bw/2, 0.999)], btype="band")
            scratch.append(arr)
            arr = np.stack([sig.filtfilt(b, a, ch).astype(np.float32) for ch in arr], axis=0)

        # 5. EQ
        if eq:
            pb = _require_pedalboard(); plugins = []
            if eq_low_db  != 0: plugins.append(pb.LowShelfFilter(cutoff_frequency_hz=eq_low_hz, gain_db=eq_low_db))
            if eq_mid_db  != 0: plugins.append(pb.PeakFilter(cutoff_frequency_hz=eq_mid_hz, gain_db=eq_mid_db, q=eq_mid_q))
            if eq_high_db != 0: plugins.append(pb.HighShelfFilter(cutoff_frequency_hz=eq_high_hz, gain_db=eq_high_db))
            if plugins:
                scratch.append(arr); arr = _apply_pedalboard(arr, sr, plugins)

        # 6. Compressor
        if compress:
            pb = _require_pedalboard()
            scratch.append(arr)
            arr = _apply_pedalboard(arr, sr, [
                pb.Compressor(threshold_db=compress_threshold, ratio=compress_ratio,
                              attack_ms=compress_attack_ms, release_ms=compress_release_ms),
                pb.Gain(gain_db=compress_makeup_db),
            ])

        # 7. Reverb
        if reverb:
            pb = _require_pedalboard()
            scratch.append(arr)
            arr = _apply_pedalboard(arr, sr, [
                pb.Reverb(room_size=reverb_room, damping=reverb_damping,
                          wet_level=reverb_wet, dry_level=reverb_dry, width=reverb_width)
            ])

        # 8. Chorus
        if chorus:
            pb = _require_pedalboard()
            scratch.append(arr)
            arr = _apply_pedalboard(arr, sr, [
                pb.Chorus(rate_hz=chorus_rate_hz, depth=chorus_depth,
                          centre_delay_ms=chorus_delay_ms, mix=chorus_mix)
            ])

        # 9. Gain / Normalize
        if normalize:
            peak = np.abs(arr).max()
            if peak > 0: scratch.append(arr); arr = (arr / peak).astype(np.float32)
        if gain_db != 0.0:
            scratch.append(arr)
            arr = np.clip(arr * (10 ** (gain_db / 20.0)), -1.0, 1.0).astype(np.float32)

        # 10. Fade
        if fade_in_sec > 0 or fade_out_sec > 0:
            n = arr.shape[-1]; env = np.ones(n, dtype=np.float32)
            fi = int(fade_in_sec * sr); fo = int(fade_out_sec * sr)
            if fi > 0: env[:fi]  *= np.linspace(0.0, 1.0, fi)
            if fo > 0: env[-fo:] *= np.linspace(1.0, 0.0, fo)
            scratch.append(arr); arr = (arr * env).astype(np.float32)

        result_audio = _to_audio(arr, sr)
        for a in scratch:
            try: del a
            except: pass
        del scratch, arr; gc.collect()

        # Preview
        ui_payload = {}
        if preview:
            try:
                import folder_paths, soundfile as sf
                p_arr, p_sr = _to_numpy(result_audio)
                fn = f"rebel_audio_{uuid.uuid4().hex[:8]}.wav"
                sf.write(os.path.join(folder_paths.get_temp_directory(), fn), p_arr.T, p_sr, subtype="PCM_16")
                del p_arr
                ui_payload = {"audio": [{"filename": fn, "subfolder": "", "type": "temp"}]}
            except Exception as e:
                print(f"[RebelAudioEdit] Preview error: {e}")

        if flush:
            _flush_memory()

        return {"ui": ui_payload, "result": (result_audio,)}


# ---------------------------------------------------------------------------
# Node: Section Extract
# ---------------------------------------------------------------------------
class RebelAudioSectionExtract:
    CATEGORY     = "RebelAudio/Section"
    RETURN_TYPES = ("AUDIO", "AUDIO", "FLOAT", "FLOAT")
    RETURN_NAMES = ("section", "original", "start_sec", "end_sec")
    FUNCTION     = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio":     ("AUDIO",),
            "start_sec": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.05}),
            "end_sec":   ("FLOAT", {"default": 2.0, "min": 0.0, "max": 3600.0, "step": 0.05}),
        }}

    def process(self, audio, start_sec, end_sec):
        arr, sr = _to_numpy(audio)
        total = arr.shape[-1] / sr
        end_sec = total if end_sec <= 0 or end_sec > total else end_sec
        if start_sec >= end_sec:
            raise ValueError("[SectionExtract] start must be < end")
        section = arr[:, int(start_sec*sr):int(end_sec*sr)].copy()
        del arr
        return (_to_audio(section, sr), audio, float(start_sec), float(end_sec))


# ---------------------------------------------------------------------------
# Node: Section Merge
# ---------------------------------------------------------------------------
class RebelAudioSectionMerge:
    CATEGORY     = "RebelAudio/Section"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION     = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "original":     ("AUDIO",),
            "section":      ("AUDIO",),
            "start_sec":    ("FLOAT", {"default": 0.0,  "min": 0.0, "max": 3600.0, "step": 0.05}),
            "end_sec":      ("FLOAT", {"default": 2.0,  "min": 0.0, "max": 3600.0, "step": 0.05}),
            "crossfade_ms": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 200.0,  "step": 5.0}),
        }}

    def process(self, original, section, start_sec, end_sec, crossfade_ms):
        orig, sr = _to_numpy(original)
        sect, _  = _to_numpy(section)
        total    = orig.shape[-1]
        s        = min(int(start_sec * sr), total)
        e        = min(int(end_sec * sr), total)
        out = np.concatenate([orig[:, :s], sect, orig[:, e:]], axis=-1).astype(np.float32)
        cf = int((crossfade_ms / 1000.0) * sr)
        if cf > 0 and s > 0:
            cl = min(cf, s, sect.shape[-1])
            fo = np.linspace(1,0,cl,dtype=np.float32); fi = np.linspace(0,1,cl,dtype=np.float32)
            out[:, s-cl:s] *= fo; out[:, s:s+cl] *= fi
            out[:, s-cl:s] += orig[:, s-cl:s] * fi
        se = s + sect.shape[-1]
        if cf > 0 and e < total:
            cl = min(cf, sect.shape[-1], total-e)
            if se+cl <= out.shape[-1]:
                fo = np.linspace(1,0,cl,dtype=np.float32); fi = np.linspace(0,1,cl,dtype=np.float32)
                out[:, se-cl:se] *= fo; out[:, se:se+cl] *= fi
                out[:, se-cl:se] += orig[:, e-cl:e] * fi
        result = _to_audio(np.clip(out,-1,1).astype(np.float32), sr)
        del orig, sect, out; gc.collect()
        return (result,)


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
