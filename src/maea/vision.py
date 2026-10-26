"""Vision module: face detection, tracking, head-pose proxy and motion."""
from __future__ import annotations

from dataclasses import dataclass, field
import cv2
import numpy as np

from .config import Config

_FRONTAL = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
_PROFILE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    frontal: bool  # True = facing camera, False = turned away (profile)

    @property
    def center(self):
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)


def _iou(a: Detection, b: Detection) -> float:
    x1, y1 = max(a.x, b.x), max(a.y, b.y)
    x2, y2 = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union else 0.0


def detect_faces(gray: np.ndarray, min_px: int = 40) -> list[Detection]:
    """Frontal faces first, then profile faces (both directions) as 'looking away'."""
    gray = _CLAHE.apply(gray)  # local contrast equalisation: robust to uneven classroom lighting
    out: list[Detection] = []
    for (x, y, w, h) in _FRONTAL.detectMultiScale(gray, 1.2, 5, minSize=(min_px, min_px)):
        out.append(Detection(int(x), int(y), int(w), int(h), True))
    width = gray.shape[1]
    profiles = list(_PROFILE.detectMultiScale(gray, 1.2, 5, minSize=(min_px, min_px)))
    flipped = cv2.flip(gray, 1)
    for (x, y, w, h) in _PROFILE.detectMultiScale(flipped, 1.2, 5, minSize=(min_px, min_px)):
        profiles.append((width - x - w, y, w, h))
    for (x, y, w, h) in profiles:
        d = Detection(int(x), int(y), int(w), int(h), False)
        if not any(_iou(d, o) > 0.3 for o in out):  # skip duplicates of frontal hits
            out.append(d)
    return out


@dataclass
class Track:
    tid: int
    center: tuple
    width: float
    last_t: float


@dataclass
class CentroidTracker:
    cfg: Config
    tracks: dict = field(default_factory=dict)
    retired: dict = field(default_factory=dict)  # seats of people who left; re-used if they return
    _next: int = 1

    def update(self, dets: list[Detection], t: float) -> list[tuple[int, Detection]]:
        assigned: list[tuple[int, Detection]] = []
        free = set(self.tracks)
        for det in sorted(dets, key=lambda d: -d.w * d.h):
            best, best_d = None, None
            for tid in free:
                tr = self.tracks[tid]
                dist = np.hypot(det.center[0] - tr.center[0], det.center[1] - tr.center[1]) / max(tr.width, 1)
                if dist <= self.cfg.max_match_dist and (best_d is None or dist < best_d):
                    best, best_d = tid, dist
            if best is None:
                best = self._reclaim(det)  # same seat as someone who stepped out? keep their ID
            if best is None:
                best = self._next
                self._next += 1
                self.tracks[best] = Track(best, det.center, det.w, t)
            elif best not in self.tracks:
                self.tracks[best] = Track(best, det.center, det.w, t)
            else:
                free.discard(best)
            tr = self.tracks[best]
            tr.center, tr.width, tr.last_t = det.center, float(det.w), t
            assigned.append((best, det))
        for tid in list(self.tracks):  # retire tracks that went missing for too long
            if t - self.tracks[tid].last_t > self.cfg.max_missing_sec:
                self.retired[tid] = self.tracks.pop(tid)
        return assigned

    def _reclaim(self, det: Detection):
        """Re-identify a returning person by seat position (classroom seats rarely change)."""
        best, best_d = None, None
        for tid, tr in self.retired.items():
            dist = np.hypot(det.center[0] - tr.center[0], det.center[1] - tr.center[1]) / max(tr.width, 1)
            if dist <= 0.8 and (best_d is None or dist < best_d):
                best, best_d = tid, dist
        if best is not None:
            del self.retired[best]
        return best


def analyse_video(path: str, cfg: Config, annotated_out: str | None = None, progress=None):
    """Return (records, meta). One record per (sampled frame, participant)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps / cfg.sample_fps)))
    tracker = CentroidTracker(cfg)
    prev_center: dict[int, tuple] = {}
    records: list[dict] = []
    writer = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            t = idx / fps
            scale = cfg.resize_width / frame.shape[1]
            small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            assigned = tracker.update(detect_faces(gray, cfg.min_face_px), t)
            for tid, d in assigned:
                px, py = prev_center.get(tid, d.center)
                motion = float(np.hypot(d.center[0] - px, d.center[1] - py) / max(d.w, 1))
                prev_center[tid] = d.center
                records.append({"t": round(t, 3), "pid": tid, "frontal": int(d.frontal), "motion": motion})
            if annotated_out:
                vis = small.copy()
                for tid, d in assigned:
                    col = (0, 200, 0) if d.frontal else (0, 165, 255)
                    cv2.rectangle(vis, (d.x, d.y), (d.x + d.w, d.y + d.h), col, 2)
                    label = f"P{tid} {'attentive' if d.frontal else 'looking away'}"
                    cv2.putText(vis, label, (d.x, max(15, d.y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(annotated_out, cv2.VideoWriter_fourcc(*"mp4v"), cfg.sample_fps, (w, h))
                writer.write(vis)
        idx += 1
        if progress and total and idx % 30 == 0:
            progress(min(idx / total, 1.0))
    cap.release()
    if idx == 0:
        raise ValueError(f"Video has no readable frames: {path}")
    if writer:
        writer.release()
    return records, {"fps": fps, "duration": idx / fps if fps else 0.0, "frames": idx}
