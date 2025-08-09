from pathlib import Path
from functools import lru_cache
from ultralytics import YOLO

# Persisted paths (update memory.md if changed)
BEST_PATH = Path("/home/naki/Desktop/yass/yolonumplate/runs_yolo/plate_det_v8n/weights/best.pt")
FALLBACK_PATH = Path("/home/naki/Desktop/yass/yolonumplate/runs_yolo/plate_det_v8n/weights/last.pt")

MODEL_WEIGHTS_PATH = BEST_PATH if BEST_PATH.exists() else FALLBACK_PATH
if not MODEL_WEIGHTS_PATH.exists():
    raise FileNotFoundError(f"No model weights found. Checked: {BEST_PATH} and {FALLBACK_PATH}")

@lru_cache(maxsize=1)
def get_model():
    return YOLO(str(MODEL_WEIGHTS_PATH))
