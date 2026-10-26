"""Tunable settings. All processing is local/offline."""
from dataclasses import dataclass


@dataclass
class Config:
    # --- vision ---
    sample_fps: float = 5.0          # frames analysed per second of video
    resize_width: int = 480          # frames are downscaled for speed
    min_face_px: int = 32            # smallest face (pixels, after resize)
    max_match_dist: float = 1.6      # tracker matching radius (in face widths)
    max_missing_sec: float = 2.0     # keep a track alive this long without detection
    # --- attendance ---
    min_presence_sec: float = 3.0    # sustained presence needed to count as attending
    min_presence_ratio: float = 0.10 # ...or this fraction of the session
    # --- audio ---
    audio_frame_ms: int = 30
    voice_margin: float = 3.0        # speech if energy > noise_floor * margin
    # --- fusion ---
    window_sec: float = 10.0         # engagement is reported per window
    w_attention: float = 0.55        # frontal-facing share of time
    w_presence: float = 0.15
    w_movement: float = 0.15
    w_audio: float = 0.15
    inactivity_still_sec: float = 20.0   # continuous stillness flagged as prolonged inactivity
    still_motion_thr: float = 0.03       # face-centre shift (in face widths/sample) below this = 'still'
    quiet_db: float = -45.0              # window mean level below this = Quiet
    noisy_db: float = -22.0              # above this = Noisy
    low_engagement: float = 0.45
    high_engagement: float = 0.75
