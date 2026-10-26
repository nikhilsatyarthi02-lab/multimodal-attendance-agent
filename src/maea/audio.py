"""Audio module: offline voice-activity / loudness analysis (no speech content is stored)."""
from __future__ import annotations

import os
import subprocess
import tempfile
import wave
import numpy as np

from .config import Config


def extract_wav(video_path: str) -> str | None:
    """Pull a mono 16 kHz wav track out of a video with ffmpeg. None if no audio / no ffmpeg."""
    out = os.path.join(tempfile.mkdtemp(), "audio.wav")
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", out],
            capture_output=True, timeout=600,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return out if r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 1000 else None


def read_wav(path: str):
    with wave.open(path, "rb") as wf:
        sr, n, ch, sw = wf.getframerate(), wf.getnframes(), wf.getnchannels(), wf.getsampwidth()
        raw = wf.readframes(n)
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[sw]
    x = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if sw == 1:
        x = (x - 128.0) / 127.0
    elif sw == 2:
        x /= 32767.0
    else:
        x /= float(np.iinfo(np.int32).max)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def analyse_audio(wav_path: str | None, cfg: Config):
    """Return per-frame arrays {t, rms, speech} or None when there is no audio."""
    if not wav_path:
        return None
    x, sr = read_wav(wav_path)
    hop = int(sr * cfg.audio_frame_ms / 1000)
    n = len(x) // hop
    if n == 0:
        return None
    frames = x[: n * hop].reshape(n, hop)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    noise_floor = max(float(np.percentile(rms, 20)), 1e-4)  # quietest 20% ~ background noise
    speech = (rms > noise_floor * cfg.voice_margin).astype(int)
    speech = (np.convolve(speech, np.ones(5) / 5, mode="same") >= 0.4).astype(int)  # drop clicks
    t = np.arange(n) * cfg.audio_frame_ms / 1000.0
    db = 20 * np.log10(rms)  # dBFS level per frame (loudness only; content is never decoded)
    return {"t": t, "rms": rms, "db": db, "speech": speech, "noise_floor": noise_floor}
