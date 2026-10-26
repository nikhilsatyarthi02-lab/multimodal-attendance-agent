"""CLI:  python main.py --video sample_data/demo.mp4"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from maea import analyze  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Multimodal Attendance & Engagement Agent (offline)")
    p.add_argument("--video", required=True, help="classroom video file")
    p.add_argument("--audio", help="optional separate .wav (default: audio track of the video)")
    p.add_argument("--out", default="outputs")
    p.add_argument("--annotate", action="store_true", help="also write annotated.mp4")
    p.add_argument("--use-gemini", action="store_true", help="optional: polish suggestions via Gemini (needs GEMINI_API_KEY)")
    a = p.parse_args()
    r = analyze(a.video, a.out, audio_path=a.audio, annotate=a.annotate, use_llm=a.use_gemini)
    s = r["summary"]
    print(f"\nDuration: {s['duration_sec']}s | participants: {s['participants_detected']} | present: {s['present']}")
    print(r["attendance"].to_string(index=False))
    print("\nFindings:")
    for f in s["findings"]:
        print(" -", f)
    print("\nSuggestions:")
    for x in s["suggestions"]:
        print(" -", x)
    print(f"\nOutputs saved in: {a.out}/")


if __name__ == "__main__":
    main()
