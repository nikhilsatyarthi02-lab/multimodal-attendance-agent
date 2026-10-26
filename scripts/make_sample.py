"""Generate a small synthetic 'classroom' clip so the project can be tried without real footage.

Uses the Grace Hopper photo bundled with matplotlib as a stand-in face, plus synthetic audio
(speech-like noise bursts). Output: sample_data/demo.mp4
"""
import os
import subprocess
import wave

import cv2
import matplotlib
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "sample_data")
os.makedirs(OUT, exist_ok=True)
W, H, FPS, DUR = 640, 480, 10, 60

face = cv2.imread(os.path.join(matplotlib.get_data_path(), "sample_data", "grace_hopper.jpg"))
face = cv2.resize(face, (220, 220))
rng = np.random.default_rng(0)


def frame(t):
    bg = np.full((H, W, 3), 70, np.uint8)
    bg += rng.integers(0, 6, bg.shape, dtype=np.uint8)
    left_here = True
    right_here = not (20 <= t < 38)           # participant 2 leaves for a while
    if left_here and not (40 <= t < 50):      # participant 1 disengages (face hidden) 40-50s
        dx, dy = int(6 * np.sin(t)), int(4 * np.cos(1.3 * t))
        bg[130 + dy:350 + dy, 60 + dx:280 + dx] = face
    if right_here:
        dx, dy = int(5 * np.cos(t)), int(3 * np.sin(0.8 * t))
        bg[130 + dy:350 + dy, 360 + dx:580 + dx] = face
    return bg


silent = os.path.join(OUT, "_silent.mp4")
vw = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
for i in range(FPS * DUR):
    vw.write(frame(i / FPS))
vw.release()

sr = 16000
t = np.arange(sr * DUR) / sr
sig = 0.003 * rng.standard_normal(len(t))                      # room noise
for a, b in [(0, 18), (42, 60)]:                                # teacher talking
    m = (t >= a) & (t < b)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t[m])            # syllable-like rhythm
    sig[m] += 0.12 * env * rng.standard_normal(m.sum())
wav = os.path.join(OUT, "demo.wav")
with wave.open(wav, "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
    wf.writeframes((np.clip(sig, -1, 1) * 32767).astype(np.int16).tobytes())

final = os.path.join(OUT, "demo.mp4")
subprocess.run(["ffmpeg", "-y", "-i", silent, "-i", wav, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest", final], capture_output=True)
if not os.path.exists(final) or os.path.getsize(final) < 1000:   # no ffmpeg: keep the silent video
    os.replace(silent, final)
else:
    os.remove(silent)
print("Created", os.path.abspath(final), "and", os.path.abspath(wav))
