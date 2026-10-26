"""End-to-end pipeline: video (+audio) -> attendance, engagement, suggestions, charts."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import audio as audio_mod
from . import vision, fusion, agent
from .config import Config


def analyze(video_path: str, out_dir: str = "outputs", cfg: Config | None = None,
            audio_path: str | None = None, annotate: bool = False,
            use_llm: bool = False, progress=None) -> dict:
    cfg = cfg or Config()
    os.makedirs(out_dir, exist_ok=True)
    annotated = os.path.join(out_dir, "annotated.mp4") if annotate else None

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    warnings: list[str] = []
    records, meta = vision.analyse_video(video_path, cfg, annotated, progress)
    audio = None
    try:  # audio is optional: any failure degrades gracefully to vision-only analysis
        wav = audio_path or audio_mod.extract_wav(video_path)
        audio = audio_mod.analyse_audio(wav, cfg)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Audio could not be analysed ({exc}); using vision signals only.")
    if audio is None and not warnings:
        warnings.append("No audio track found; using vision signals only.")
    if not records:
        warnings.append("No faces detected; check camera angle, lighting and face size.")

    attendance = fusion.build_attendance(records, meta["duration"], cfg)
    expected = {int(x[1:]) for x in attendance.loc[attendance["status"] == "Present", "participant"]}
    kept = [r for r in records if r["pid"] in expected] if expected else records  # drop flickers
    timeline, per_part = fusion.build_engagement(kept, audio, meta["duration"], cfg, expected)
    students = fusion.build_student_summary(kept, attendance, per_part, cfg)
    insights = agent.reason(timeline, per_part, cfg, students)
    if use_llm:
        insights = agent.refine_with_gemini(insights)

    _make_browser_playable(annotated, warnings)
    attendance.to_csv(os.path.join(out_dir, "attendance.csv"), index=False)
    timeline.to_csv(os.path.join(out_dir, "engagement_timeline.csv"), index=False)
    per_part.to_csv(os.path.join(out_dir, "participant_sessions.csv"), index=False)
    students.to_csv(os.path.join(out_dir, "student_session_analytics.csv"), index=False)
    chart = _chart(timeline, os.path.join(out_dir, "engagement.png"), cfg)
    summary = {"video": os.path.basename(video_path), "duration_sec": round(meta["duration"], 1),
               "audio_available": audio is not None, "participants_detected": len(attendance),
               "present": int((attendance["status"] == "Present").sum()) if len(attendance) else 0,
               "warnings": warnings, **insights}
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return {"summary": summary, "attendance": attendance, "timeline": timeline,
            "participants": per_part, "students": students, "chart": chart, "annotated": annotated}


def _make_browser_playable(path: str, warnings: list) -> None:
    """OpenCV writes MPEG-4 Part 2, which browsers cannot play. Re-encode to H.264 with ffmpeg if present."""
    import shutil
    import subprocess
    if not (path and os.path.exists(path)):
        return
    if not shutil.which("ffmpeg"):
        warnings.append("Annotated video saved, but ffmpeg is missing so it may not play in the browser. "
                        "Install ffmpeg (winget install ffmpeg) or open outputs/annotated.mp4 in VLC.")
        return
    tmp = path + ".h264.mp4"
    r = subprocess.run(["ffmpeg", "-y", "-i", path, "-vcodec", "libx264", "-pix_fmt", "yuv420p", tmp],
                       capture_output=True)
    if r.returncode == 0 and os.path.exists(tmp):
        os.replace(tmp, path)


def _chart(timeline, path, cfg):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(timeline["t_mid"] / 60, timeline["engagement"], marker="o", color="#2563eb", label="Engagement")
    if timeline["audio_activity"].notna().any():
        ax.plot(timeline["t_mid"] / 60, timeline["audio_activity"].astype(float), color="#f59e0b",
                linestyle="--", label="Audio activity")
    ax.axhline(cfg.low_engagement, color="#dc2626", lw=1, ls=":", label="Low threshold")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Session time (min)")
    ax.set_ylabel("Score (0-1)")
    ax.set_title("Class engagement over time")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
