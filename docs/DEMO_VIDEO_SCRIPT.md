# Demo video script (about 4 minutes) - required for mentor review

Record your screen (Win+G, OBS, or Zoom "record") with your voice. Upload as an **unlisted YouTube** or
Google Drive ("anyone with the link") video and paste that link as the **Demo Link** in SkillWallet.

1. **0:00 Intro (20 s)** - "I'm Nikhil Satyarthi. This is the Multimodal Attendance & Engagement Agent: an
   offline, privacy-preserving classroom intelligence system using computer vision, audio signal processing
   and agent-based reasoning."
2. **0:20 Project structure (30 s)** - show the repo folders: `vision.py`, `audio.py`, `fusion.py`, `agent.py`, `app.py`.
3. **0:50 Setup (20 s)** - run `python scripts/check_env.py` (Epic 1).
4. **1:10 Run core app (40 s)** - `python main.py --video sample_data/demo.mp4 --annotate`; show the console
   attendance table, findings and suggestions (Epics 2-5).
5. **1:50 Dashboard (80 s)** - `streamlit run app.py`; click *Analyze session*; walk through the tabs:
   Attendance (sustained presence), Engagement (chart, noise level), Student sessions (feedback), AI suggestions.
   Upload **your own real classroom clip** here if you have one.
6. **3:10 Annotated video (20 s)** - show green "attentive" and orange "looking away" boxes.
7. **3:30 Validation (20 s)** - open `docs/validation_report.md` (9/9 edge cases) and run `pytest`.
8. **3:50 Conclusion (10 s)** - assistive, offline, ethical; future work: deep head-pose models.
