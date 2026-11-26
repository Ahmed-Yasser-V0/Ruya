"""Every AI model is loaded ONCE here and shared by all modes.

Before this, yolov8m.pt was loaded three times (app, detection, finder), which
tripled memory use and slowed start-up. A single lock also makes sure two
requests never run the same model at the same time (Flask is multi-threaded).
"""
import os
import threading

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFERENCE_LOCK = threading.RLock()

_cache = {}
_cache_lock = threading.Lock()


def _load(key, loader):
    if key not in _cache:
        with _cache_lock:
            if key not in _cache:
                _cache[key] = loader()
    return _cache[key]


def path(*parts):
    return os.path.join(BASE_DIR, *parts)


def object_model():
    from ultralytics import YOLO
    return _load("yolo", lambda: YOLO(path("yolov8m.pt")))


def currency_model():
    from ultralytics import YOLO
    return _load("currency", lambda: YOLO(path("currency.pt")))


def open_vocab_model():
    """YOLOE with a built-in vocabulary of everyday things blind users ask for
    (door, stairs, keys, glasses, wallet, medicine...). The vocabulary is stored
    inside models/glasses_vocab.pt, so no text model or internet is needed.
    Returns None if the file is missing (the app then works without it)."""
    def load():
        file = path("models", "glasses_vocab.pt")
        if not os.path.exists(file):
            return None
        from ultralytics import YOLOE
        return YOLOE(file)
    return _load("open_vocab", load)
