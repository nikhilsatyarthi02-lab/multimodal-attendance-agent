"""Verify that all required libraries (and optional ffmpeg) are available."""
import importlib
import shutil

ok = True
for mod in ["cv2", "numpy", "pandas", "scipy", "matplotlib", "streamlit"]:
    try:
        m = importlib.import_module(mod)
        print(f"[ OK ] {mod} {getattr(m, '__version__', '')}")
    except ImportError:
        ok = False
        print(f"[MISS] {mod}  ->  pip install -r requirements.txt")
print("[ OK ] ffmpeg found" if shutil.which("ffmpeg") else
      "[WARN] ffmpeg not found - audio track cannot be extracted (vision-only mode still works)")
print("Environment ready." if ok else "Fix the missing packages above.")
