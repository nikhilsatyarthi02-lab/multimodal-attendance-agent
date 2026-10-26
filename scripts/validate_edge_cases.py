"""Activity 6.3 - validate outputs and edge cases. Writes docs/validation_report.md."""
import os
import sys
import tempfile
import wave

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from maea import analyze  # noqa: E402

FULL = os.path.join(ROOT, "sample_data", "demo.mp4")
TMP = tempfile.mkdtemp()
DEMO = FULL
if os.path.exists(FULL):  # use a 15 s excerpt so the whole suite runs in about a minute
    import subprocess
    short_demo = os.path.join(TMP, "demo15.mp4")
    r = subprocess.run(["ffmpeg", "-y", "-i", FULL, "-t", "15", short_demo], capture_output=True)
    if r.returncode == 0:
        DEMO = short_demo
OUT = os.path.join(TMP, "out")


def write_video(name, frames_fn, n, fps=10, size=(640, 480)):
    path = os.path.join(TMP, name)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n):
        vw.write(frames_fn(i))
    vw.release()
    return path


def write_wav(name, x, sr=16000):
    path = os.path.join(TMP, name)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    return path


def darken(src, factor):
    cap = cv2.VideoCapture(src)
    path = os.path.join(TMP, "dark.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS), (640, 480))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        vw.write(cv2.resize((f * factor).astype(np.uint8), (640, 480)))
    vw.release()
    return path


def expect_error(fn, kinds):
    try:
        fn()
    except kinds as e:
        return True, f"raised {type(e).__name__} as intended"
    except Exception as e:  # noqa: BLE001
        return False, f"unexpected {type(e).__name__}: {e}"
    return False, "no error raised"


def expect_ok(fn, check=lambda r: True):
    try:
        r = fn()
    except Exception as e:  # noqa: BLE001
        return False, f"crashed: {type(e).__name__}: {e}"
    ok = check(r)
    s = r["summary"]
    return ok, f"completed; participants={s['participants_detected']}, present={s['present']}, warnings={s['warnings']}"


cases = []
blank = lambda i: np.full((480, 640, 3), 70, np.uint8)  # noqa: E731

cases.append(("Missing video file", *expect_error(lambda: analyze("nope.mp4", OUT), FileNotFoundError)))
bad = os.path.join(TMP, "bad.mp4"); open(bad, "wb").write(os.urandom(5000))
cases.append(("Corrupted / non-video file", *expect_error(lambda: analyze(bad, OUT), (FileNotFoundError, ValueError))))
black = write_video("black.mp4", blank, 50)
cases.append(("No faces in video (empty room)", *expect_ok(lambda: analyze(black, OUT),
             lambda r: r["summary"]["present"] == 0 and any("No faces" in w for w in r["summary"]["warnings"]))))
cases.append(("Video without audio track", *expect_ok(lambda: analyze(black, OUT),
             lambda r: not r["summary"]["audio_available"])))
short = write_video("short.mp4", blank, 5)
cases.append(("Very short video (0.5 s)", *expect_ok(lambda: analyze(short, OUT))))
if os.path.exists(DEMO):
    silent = write_wav("silent.wav", np.zeros(16000 * 15))
    cases.append(("Completely silent audio", *expect_ok(lambda: analyze(DEMO, OUT, audio_path=silent))))
    noisy = write_wav("noisy.wav", 0.4 * np.random.default_rng(0).standard_normal(16000 * 15))
    cases.append(("Constant loud noise", *expect_ok(lambda: analyze(DEMO, OUT, audio_path=noisy),
                 lambda r: (r["timeline"]["noise_level"] == "Noisy").any())))
    dark = darken(DEMO, 0.35)
    cases.append(("Low lighting (35% brightness)", *expect_ok(lambda: analyze(dark, OUT))))
    cases.append(("Normal demo session (reference)", *expect_ok(lambda: analyze(DEMO, OUT),
                 lambda r: r["summary"]["present"] == 2)))

lines = ["# Validation report (Activity 6.3)", "", "| # | Case | Result | Detail |", "|---|------|--------|--------|"]
for i, (name, ok, detail) in enumerate(cases, 1):
    lines.append(f"| {i} | {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
passed = sum(1 for c in cases if c[1])
lines += ["", f"**{passed}/{len(cases)} cases passed.**"]
os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
with open(os.path.join(ROOT, "docs", "validation_report.md"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
sys.exit(0 if passed == len(cases) else 1)
