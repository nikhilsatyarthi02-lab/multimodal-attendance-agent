"""Fusion engine: attendance decisions, multimodal signal fusion and explainable engagement scoring."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def fmt_t(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


# ---------------------------------------------------------------- attendance (Activity 2.2)
def build_attendance(records: list[dict], duration: float, cfg: Config) -> pd.DataFrame:
    """Attendance by *sustained* presence (not a single-frame hit)."""
    cols = ["participant", "first_seen", "last_seen", "present_sec", "presence_pct", "status"]
    if not records:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(records)
    dt = 1.0 / cfg.sample_fps
    rows = []
    for pid, g in df.groupby("pid"):
        ts = np.sort(g["t"].values)
        present = dt + np.minimum(np.diff(ts), cfg.max_missing_sec).sum() if len(ts) > 1 else dt
        pct = present / duration if duration else 0.0
        ok = present >= cfg.min_presence_sec or pct >= cfg.min_presence_ratio
        rows.append({
            "participant": f"P{pid}", "first_seen": fmt_t(ts[0]), "last_seen": fmt_t(ts[-1]),
            "present_sec": round(float(present), 1), "presence_pct": round(100 * pct, 1),
            "status": "Present" if ok else "Absent (not sustained)",
        })
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------- audio features (Activity 3.2)
def noise_level(db: float, cfg: Config) -> str:
    return "Quiet" if db < cfg.quiet_db else ("Noisy" if db > cfg.noisy_db else "Moderate")


def audio_windows(audio: dict | None, n_win: int, cfg: Config):
    """Per window: speech-activity ratio and mean level (dBFS)."""
    speech = np.full(n_win, np.nan)
    level = np.full(n_win, np.nan)
    if audio is not None:
        for w in range(n_win):
            m = (audio["t"] >= w * cfg.window_sec) & (audio["t"] < (w + 1) * cfg.window_sec)
            if m.any():
                speech[w] = audio["speech"][m].mean()
                level[w] = 20 * np.log10(np.sqrt((audio["rms"][m] ** 2).mean()) + 1e-12)
    return speech, level


# ---------------------------------------------------------------- stillness helper (inactivity)
def longest_still_sec(g: pd.DataFrame, cfg: Config) -> float:
    """Longest continuous stretch with (almost) no movement, in seconds."""
    g = g.sort_values("t")
    best = cur = 0.0
    prev_t = None
    for t, m in zip(g["t"].values, g["motion"].values):
        if m < cfg.still_motion_thr and (prev_t is None or t - prev_t <= cfg.max_missing_sec):
            cur += (t - prev_t) if prev_t is not None else 0.0
        else:
            cur = 0.0
        best = max(best, cur)
        prev_t = t
    return round(best, 1)


# ---------------------------------------------------------------- fusion + scoring (Activities 4.2, 4.3)
def score_components(g: pd.DataFrame, speech_ratio: float, exp_frames: float, cfg: Config) -> dict:
    """Four interpretable components in 0-1; the final score is their weighted sum."""
    attention = float(g["frontal"].mean())                 # vision: facing the camera
    presence = min(len(g) / exp_frames, 1.0)               # vision: share of window present
    still_ratio = float((g["motion"] < cfg.still_motion_thr).mean())
    mv = float(g["motion"].mean())
    if mv > 0.35:                                          # restless / very large movement
        movement = 0.4
    else:                                                  # mostly still = low activity
        movement = 1.0 - 0.7 * max(0.0, still_ratio - 0.5) / 0.5
    audio_c = 0.5 if np.isnan(speech_ratio) else float(np.clip(speech_ratio / 0.5, 0, 1))  # audio: interaction
    score = (cfg.w_attention * attention + cfg.w_presence * presence
             + cfg.w_movement * movement + cfg.w_audio * audio_c)
    return {"attention": attention, "presence": presence, "movement": movement, "audio": audio_c,
            "still_ratio": still_ratio, "motion": mv, "engagement": float(score)}


def explain(c: dict, cfg: Config) -> str:
    """One human-readable sentence saying *why* the score is what it is."""
    parts = []
    parts.append("mostly facing the camera" if c["attention"] >= 0.75 else
                 "often looking away" if c["attention"] >= 0.4 else "rarely facing the camera")
    if c["presence"] < 0.6:
        parts.append(f"in frame only {c['presence']:.0%} of the window")
    if c["still_ratio"] > 0.8:
        parts.append("very little movement")
    elif c["motion"] > 0.35:
        parts.append("restless movement")
    parts.append("class audio active" if c["audio"] >= 0.7 else "little audio interaction" if c["audio"] < 0.3 else "some audio interaction")
    return "; ".join(parts)


def build_engagement(records: list[dict], audio: dict | None, duration: float, cfg: Config,
                     expected: set | None = None):
    """Return (class_timeline, per_participant_windows) DataFrames.

    `expected` = participant IDs counted as attending; anyone expected but missing from a
    window contributes a score of 0 (so people leaving the room lowers class engagement).
    """
    n_win = max(1, int(np.ceil(duration / cfg.window_sec)))
    vdf = pd.DataFrame(records) if records else pd.DataFrame(columns=["t", "pid", "frontal", "motion"])
    exp_frames = cfg.sample_fps * cfg.window_sec
    speech, level = audio_windows(audio, n_win, cfg)

    part_rows, class_rows = [], []
    for w in range(n_win):
        t0, t1 = w * cfg.window_sec, (w + 1) * cfg.window_sec
        wdf = vdf[(vdf["t"] >= t0) & (vdf["t"] < t1)]
        scores, seen = [], set()
        for pid, g in wdf.groupby("pid"):
            c = score_components(g, speech[w], exp_frames, cfg)
            part_rows.append({"window": w, "start": fmt_t(t0), "participant": f"P{pid}",
                              "attention": round(c["attention"], 2), "presence": round(c["presence"], 2),
                              "movement_score": round(c["movement"], 2), "audio_score": round(c["audio"], 2),
                              "motion": round(c["motion"], 3), "engagement": round(c["engagement"], 3),
                              "explanation": explain(c, cfg)})
            seen.add(pid)
            if not expected or pid in expected:
                scores.append(c["engagement"])
        if expected:
            scores += [0.0] * len(expected - seen)
        gone = sorted(expected - seen) if expected else []
        reason = ""
        if gone:
            reason = "Not in frame: " + ", ".join(f"P{p}" for p in gone)
        class_rows.append({
            "window": w, "start": fmt_t(t0), "t_mid": (t0 + t1) / 2, "participants": len(seen),
            "audio_activity": None if np.isnan(speech[w]) else round(float(speech[w]), 2),
            "noise_db": None if np.isnan(level[w]) else round(float(level[w]), 1),
            "noise_level": "n/a" if np.isnan(level[w]) else noise_level(float(level[w]), cfg),
            "engagement": round(float(np.mean(scores)), 3) if scores else 0.0,
            "note": reason,
        })
    return pd.DataFrame(class_rows), pd.DataFrame(part_rows)


# ---------------------------------------------------------------- student session analytics (Activity 5.2)
def build_student_summary(records, attendance, per_part, cfg: Config) -> pd.DataFrame:
    cols = ["participant", "status", "presence_pct", "avg_attention", "avg_engagement",
            "best_window", "lowest_window", "low_windows", "longest_still_sec", "feedback"]
    if attendance.empty:
        return pd.DataFrame(columns=cols)
    vdf = pd.DataFrame(records)
    rows = []
    for _, a in attendance.iterrows():
        pid = int(a["participant"][1:])
        d = per_part[per_part["participant"] == a["participant"]]
        still = longest_still_sec(vdf[vdf["pid"] == pid], cfg) if len(vdf) else 0.0
        if d.empty:
            continue
        best, worst = d.loc[d["engagement"].idxmax()], d.loc[d["engagement"].idxmin()]
        low = int((d["engagement"] < cfg.low_engagement).sum())
        avg = float(d["engagement"].mean())
        if a["status"] != "Present":
            tip = "Not counted as attending (presence was too brief to be reliable)."
        elif avg >= cfg.high_engagement and low == 0:
            tip = "Consistently engaged - keep it up."
        elif still >= cfg.inactivity_still_sec:
            tip = f"Long still stretch (~{still:.0f}s). Try taking notes or asking questions to stay active."
        elif low:
            tip = f"Attention dipped in {low} window(s), e.g. near {worst['start']}. Notice what pulls your focus away."
        else:
            tip = "Generally engaged with small dips - a good base to build on."
        rows.append({"participant": a["participant"], "status": a["status"], "presence_pct": a["presence_pct"],
                     "avg_attention": round(float(d["attention"].mean()), 2), "avg_engagement": round(avg, 3),
                     "best_window": best["start"], "lowest_window": worst["start"], "low_windows": low,
                     "longest_still_sec": still, "feedback": tip})
    return pd.DataFrame(rows, columns=cols)
