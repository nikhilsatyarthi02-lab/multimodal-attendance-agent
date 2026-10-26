"""Agent: reasons over the engagement timeline and produces assistive teaching suggestions.

Runs fully offline with a rule-based reasoner. Optionally, if GEMINI_API_KEY is set and
use_llm=True, the suggestions are polished by Gemini (cloud call; disabled by default).
"""
from __future__ import annotations

import json
import os
import urllib.request

import numpy as np
import pandas as pd

from .config import Config


def reason(timeline: pd.DataFrame, participants: pd.DataFrame, cfg: Config,
           students: pd.DataFrame | None = None) -> dict:
    """Rule-based, explainable reasoning over fused signals -> findings + suggestions.

    Every finding states the evidence behind it; nothing is enforced - suggestions are assistive.
    """
    findings, suggestions = [], []
    live = timeline[timeline["participants"] > 0]
    if live.empty:
        return {"findings": ["No participants were detected."], "suggestions": [
            "Check camera angle/lighting so faces are visible, then re-run the analysis."]}

    eng = live["engagement"].values
    mean, std = float(eng.mean()), float(eng.std())
    findings.append(f"Average session engagement: {mean:.0%}.")

    # attention drops: below the low threshold or well under the session mean
    low = (live["engagement"] < cfg.low_engagement) | (live["engagement"] < mean - 1.0 * std - 1e-9)
    runs = _runs(low.values)
    for a, b in runs:
        if True:  # every window is already a smoothed 10 s average, so even one low window matters
            s, e = live.iloc[a]["start"], live.iloc[b]["start"]
            why = live.iloc[a]["note"] or _top_reason(participants, live.iloc[a]["window"])
            findings.append(f"Attention drop at {s}" + (f" to {e}" if b > a else "") + f" ({b - a + 1} window(s))"
                            + (f" - {why}" if why else "") + ".")
            suggestions.append(f"Around {s}, insert a quick poll, question or 2-minute activity to re-engage the class.")

    # interaction peaks
    peaks = live[live["engagement"] >= cfg.high_engagement]
    if len(peaks):
        best = live.loc[live["engagement"].idxmax()]
        findings.append(f"Engagement peaked at {best['start']} ({best['engagement']:.0%}).")
        suggestions.append(f"Reuse the approach used near {best['start']} (it held attention best).")

    # pacing: long stretches with almost no audio activity
    au = live["audio_activity"].dropna()
    if len(au):
        quiet = _runs((live["audio_activity"].fillna(1.0) < 0.05).values)
        if any(b - a + 1 >= 3 for a, b in quiet):
            findings.append("Long silent stretch detected (3+ windows with almost no audio).")
            suggestions.append("Break long silent periods with discussion or Q&A so learners can interact.")
        if float(au.mean()) > 0.85:
            findings.append("Audio is almost continuous (very little pause time).")
            suggestions.append("Add short pauses between concepts to let learners absorb and ask questions.")

    # classroom noise (audio level is a noise signal only - no speech content is analysed)
    if "noise_level" in live and (live["noise_level"] == "Noisy").sum() >= 2:
        findings.append("Noisy classroom conditions in several windows.")
        suggestions.append("Use a signal to settle the room (or structured group work) when noise rises.")

    # prolonged inactivity (Scenario 2: inactivity duration)
    if students is not None and len(students):
        idle = students[(students["status"] == "Present") & (students["longest_still_sec"] >= cfg.inactivity_still_sec)]
        if len(idle):
            findings.append(f"Prolonged inactivity (>= {cfg.inactivity_still_sec:.0f}s still) for: "
                            + ", ".join(idle["participant"]) + ".")
            suggestions.append("Add a hands-on step or quick question so passive listeners participate.")

    # declining trend
    if len(eng) >= 4:
        slope = float(np.polyfit(np.arange(len(eng)), eng, 1)[0])
        if slope < -0.01:
            findings.append("Engagement trends downward across the session.")
            suggestions.append("Consider shorter segments or a mid-session break to counter fatigue.")

    # individual follow-up (aggregate, never punitive)
    if len(participants):
        avg = participants.groupby("participant")["engagement"].mean()
        low_p = avg[avg < cfg.low_engagement].index.tolist()
        if low_p:
            findings.append(f"Lower average engagement for: {', '.join(low_p)}.")
            suggestions.append("Check in informally with these learners; the system supports, not judges.")

    if not suggestions:
        suggestions.append("Engagement was steady - keep the current pacing and structure.")
    return {"findings": findings, "suggestions": suggestions}


def _top_reason(participants: pd.DataFrame, window: int) -> str:
    """Explain a drop using the weakest participant explanation in that window."""
    d = participants[participants["window"] == window]
    return d.loc[d["engagement"].idxmin(), "explanation"] if len(d) else ""


def _runs(mask) -> list[tuple[int, int]]:
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def refine_with_gemini(result: dict, model: str = "gemini-1.5-flash") -> dict:
    """Optional: ask Gemini to rewrite suggestions more naturally. Falls back silently."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return result
    prompt = ("You assist a teacher. Rewrite these findings into 3-5 short, practical, non-judgmental "
              "teaching suggestions. Return plain bullet lines.\n" + json.dumps(result))
    try:
        req = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
            data=json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            text = json.load(r)["candidates"][0]["content"]["parts"][0]["text"]
        lines = [l.lstrip("-* ").strip() for l in text.splitlines() if l.strip()]
        return {**result, "suggestions": lines or result["suggestions"], "llm_refined": True}
    except Exception:
        return result
