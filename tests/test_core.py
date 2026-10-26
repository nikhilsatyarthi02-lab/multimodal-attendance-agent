import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from maea import agent, audio, fusion  # noqa: E402
from maea.config import Config  # noqa: E402

CFG = Config()


def _records(pid, t0, t1, frontal=1, fps=5):
    return [{"t": round(t, 3), "pid": pid, "frontal": frontal, "motion": 0.05} for t in np.arange(t0, t1, 1 / fps)]


def test_sustained_presence_vs_flicker():
    rec = _records(1, 0, 30) + _records(2, 10, 10.2)  # P2 appears for a single frame
    att = fusion.build_attendance(rec, 30, CFG).set_index("participant")
    assert att.loc["P1", "status"] == "Present"
    assert att.loc["P2", "status"].startswith("Absent")


def test_no_records_gives_empty_attendance():
    assert fusion.build_attendance([], 30, CFG).empty


def test_engagement_drop_when_person_leaves():
    rec = _records(1, 0, 40) + _records(2, 0, 20)
    tl, _ = fusion.build_engagement(rec, None, 40, CFG, expected={1, 2})
    assert tl.loc[0, "engagement"] > tl.loc[3, "engagement"]


def test_looking_away_lowers_attention():
    good = fusion.build_engagement(_records(1, 0, 10, frontal=1), None, 10, CFG, {1})[0]["engagement"][0]
    away = fusion.build_engagement(_records(1, 0, 10, frontal=0), None, 10, CFG, {1})[0]["engagement"][0]
    assert good > away


def test_voice_activity_detection():
    sr = 16000
    rng = np.random.default_rng(1)
    x = 0.002 * rng.standard_normal(sr * 4)
    x[sr * 2:] += 0.2 * rng.standard_normal(sr * 2)  # loud second half
    import tempfile, wave
    p = os.path.join(tempfile.mkdtemp(), "t.wav")
    with wave.open(p, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes((x * 32767).astype(np.int16).tobytes())
    a = audio.analyse_audio(p, CFG)
    half = len(a["speech"]) // 2
    assert a["speech"][:half].mean() < 0.2 < 0.8 < a["speech"][half + 5:].mean()


def test_agent_flags_drop_and_suggests():
    rec = _records(1, 0, 60)
    tl, pp = fusion.build_engagement(rec, None, 60, CFG, {1})
    tl.loc[2, "engagement"] = 0.1
    out = agent.reason(tl, pp, CFG)
    assert any("drop" in f.lower() for f in out["findings"])
    assert out["suggestions"]


def test_explanation_is_human_readable():
    rec = _records(1, 0, 10, frontal=0)
    _, pp = fusion.build_engagement(rec, None, 10, CFG, {1})
    assert "looking away" in pp.loc[0, "explanation"] or "rarely" in pp.loc[0, "explanation"]


def test_student_summary_and_inactivity():
    rec = [{"t": round(t, 3), "pid": 1, "frontal": 1, "motion": 0.0} for t in np.arange(0, 40, 0.2)]
    att = fusion.build_attendance(rec, 40, CFG)
    _, pp = fusion.build_engagement(rec, None, 40, CFG, {1})
    st = fusion.build_student_summary(rec, att, pp, CFG)
    assert st.loc[0, "longest_still_sec"] >= CFG.inactivity_still_sec
    assert st.loc[0, "feedback"]


def test_noise_levels():
    assert fusion.noise_level(-60, CFG) == "Quiet"
    assert fusion.noise_level(-10, CFG) == "Noisy"


def test_pipeline_handles_missing_and_empty_video():
    import tempfile
    import cv2
    from maea import analyze
    try:
        analyze("does_not_exist.mp4", tempfile.mkdtemp())
        assert False
    except FileNotFoundError:
        pass
    p = os.path.join(tempfile.mkdtemp(), "blank.mp4")
    vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for _ in range(20):
        vw.write(np.full((240, 320, 3), 80, np.uint8))
    vw.release()
    r = analyze(p, tempfile.mkdtemp())
    assert r["summary"]["present"] == 0 and r["summary"]["warnings"]
