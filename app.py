"""Streamlit dashboard:  streamlit run app.py"""
import os
import sys
import tempfile

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from maea import analyze  # noqa: E402
from maea.config import Config  # noqa: E402

st.set_page_config(page_title="Multimodal Attendance & Engagement Agent", layout="wide")
st.title("Multimodal Attendance & Engagement Agent")
st.caption("Offline, privacy-preserving classroom analytics: computer vision + audio signals + agent reasoning. "
           "No faces or audio are stored or sent anywhere.")

with st.sidebar:
    st.header("Settings")
    window = st.slider("Engagement window (sec)", 5, 60, 10, 5)
    min_presence = st.slider("Min. sustained presence (sec)", 1, 15, 3)
    annotate = st.checkbox("Create annotated video", value=True)
    use_llm = st.checkbox("Polish suggestions with Gemini (optional, needs GEMINI_API_KEY)", value=False)

up = st.file_uploader("Upload a classroom video (mp4, avi, mov)", type=["mp4", "avi", "mov", "mkv"])
sample = os.path.join(os.path.dirname(__file__), "sample_data", "demo.mp4")
use_sample = False
if up is None and os.path.exists(sample):
    use_sample = st.checkbox("Use the bundled sample video", value=True)

if (up is not None or use_sample) and st.button("Analyze session", type="primary"):
    if up is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1])
        tmp.write(up.read())
        tmp.close()
        path = tmp.name
    else:
        path = sample
    cfg = Config(window_sec=float(window), min_presence_sec=float(min_presence))
    bar = st.progress(0.0, text="Analysing video...")
    res = analyze(path, "outputs", cfg, annotate=annotate, use_llm=use_llm,
                  progress=lambda p: bar.progress(p, text="Analysing video..."))
    bar.empty()
    st.session_state["res"] = res

res = st.session_state.get("res")
if res:
    s = res["summary"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration", f"{s['duration_sec']:.0f}s")
    c2.metric("Participants detected", s["participants_detected"])
    c3.metric("Marked present", s["present"])
    c4.metric("Avg engagement", f"{res['timeline']['engagement'].mean():.0%}")
    for w in s.get("warnings", []):
        st.warning(w)

    tab1, tab2, tab3, tab4 = st.tabs(["Attendance", "Engagement", "Student sessions", "AI suggestions"])
    with tab1:
        st.dataframe(res["attendance"], use_container_width=True, hide_index=True)
        st.download_button("Download attendance.csv", res["attendance"].to_csv(index=False), "attendance.csv")
    with tab2:
        st.image(res["chart"])
        st.line_chart(res["timeline"].set_index("start")[["engagement"]])
        st.caption("Noise level is loudness only (Quiet / Moderate / Noisy) - speech content is never analysed.")
        st.dataframe(res["timeline"].drop(columns=["t_mid"]), use_container_width=True, hide_index=True)
    with tab3:
        st.subheader("Student session analytics")
        st.dataframe(res["students"], use_container_width=True, hide_index=True)
        st.download_button("Download student_session_analytics.csv", res["students"].to_csv(index=False),
                           "student_session_analytics.csv")
        who = st.selectbox("Participant", sorted(res["participants"]["participant"].unique()) or ["-"])
        d = res["participants"][res["participants"]["participant"] == who]
        st.line_chart(d.set_index("start")[["engagement", "attention"]])
        st.dataframe(d[["start", "attention", "presence", "movement_score", "audio_score", "engagement", "explanation"]],
                     use_container_width=True, hide_index=True)
    with tab4:
        st.subheader("Findings")
        for f in s["findings"]:
            st.write("-", f)
        st.subheader("Suggestions for the instructor")
        for x in s["suggestions"]:
            st.write("-", x)
        st.caption("These are assistive suggestions only; the instructor makes the final call.")
    if res.get("annotated") and os.path.exists(res["annotated"]):
        with st.expander("Annotated video"):
            st.video(res["annotated"])
            with open(res["annotated"], "rb") as fh:
                st.download_button("Download annotated.mp4 (play in VLC if the player above is blank)",
                                   fh.read(), "annotated.mp4", mime="video/mp4")
