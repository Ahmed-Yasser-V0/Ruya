"""Mood (facial expression) and approximate age group of a face.

Models (run with OpenCV, no extra libraries, files in models/):
  - Facial expression: OpenCV Zoo MobileFaceNet, Apache-2.0.
    7 expressions, 88.3% accuracy on RAF-DB (published).
  - Age group: Levi & Hassner GoogLeNet trained on Adience (ONNX Model Zoo, Apache-2.0 files).
    8 age groups. Published exact-group accuracy on Adience is about 50%, and about 85% within
    one neighbouring group, so the app says "about 25 to 32" and votes over several frames.
    The Adience photos were collected for research; check the licence before selling this part.

Both need the 5 face landmarks that YuNet already gives, so they add only ~10 ms per face.
"""
import os

import cv2
import numpy as np

from core.models import INFERENCE_LOCK, path

FER_PATH = path("models", "facial_expression_recognition_mobilefacenet_2022july.onnx")
AGE_PATH = path("models", "age_googlenet.onnx")

EXPRESSIONS = ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
# what a listener / an operator understands, and the "sentiment" side of it
MOOD = {"angry": ("angry", "negative"), "disgust": ("disgusted", "negative"),
        "fearful": ("scared", "negative"), "happy": ("happy", "positive"),
        "neutral": ("calm", "neutral"), "sad": ("sad", "negative"),
        "surprised": ("surprised", "neutral")}
MOOD_AR = {"angry": "متضايق", "disgusted": "قرفان", "scared": "خايف", "happy": "مبسوط",
           "calm": "هادي", "sad": "زعلان", "surprised": "مستغرب"}
AGE_GROUPS = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60+"]
MIN_MOOD_CONF = 0.45          # below this the expression is reported as calm

# reference positions of the 5 landmarks in a 112x112 aligned face (OpenCV Zoo)
_REF = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                 [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
_nets = {}


def available():
    return os.path.exists(FER_PATH) and os.path.exists(AGE_PATH)


def _net(file):
    if file not in _nets:
        _nets[file] = cv2.dnn.readNet(file)
    return _nets[file]


def _softmax(x):
    x = np.asarray(x, dtype=np.float64).flatten()
    if x.min() >= 0 and abs(x.sum() - 1) < 1e-3:      # already probabilities
        return x
    e = np.exp(x - x.max())
    return e / e.sum()


def mood(image, face):
    """face = YuNet row. Returns (expression, probability)."""
    landmarks = np.asarray(face[4:14], dtype=np.float32).reshape(5, 2)
    matrix, _ = cv2.estimateAffinePartial2D(landmarks, _REF, method=cv2.LMEDS)
    if matrix is None:
        return None, 0.0
    crop = cv2.warpAffine(image, matrix, (112, 112))
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = cv2.dnn.blobFromImage((rgb - 0.5) / 0.5)
    with INFERENCE_LOCK:
        net = _net(FER_PATH)
        net.setInput(blob, "data")
        out = net.forward("label")
    p = _softmax(out[0])
    i = int(p.argmax())
    return EXPRESSIONS[i], float(p[i])


def age_scores(image, face):
    """Probabilities over AGE_GROUPS for one face (face crop with a margin, as in training)."""
    h, w = image.shape[:2]
    x, y, fw, fh = [float(v) for v in face[:4]]
    m = 0.4 * max(fw, fh)
    x1, y1 = int(max(0, x - m)), int(max(0, y - m))
    x2, y2 = int(min(w, x + fw + m)), int(min(h, y + fh + m))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    blob = cv2.dnn.blobFromImage(cv2.resize(crop, (224, 224)), 1.0, (224, 224), (104, 117, 123), swapRB=False)
    with INFERENCE_LOCK:
        net = _net(AGE_PATH)
        net.setInput(blob)
        out = net.forward()
    return _softmax(out[0])


def analyse(image, face):
    """{'expression', 'mood', 'sentiment', 'mood_conf', 'age_group', 'age_scores'} or {} if unavailable."""
    if not available():
        return {}
    try:
        expression, conf = mood(image, face)
        scores = age_scores(image, face)
    except cv2.error:
        return {}
    result = {}
    if expression:
        if conf < MIN_MOOD_CONF:
            expression = "neutral"
        word, sentiment = MOOD[expression]
        result.update(expression=expression, mood=word, sentiment=sentiment, mood_conf=round(conf, 2))
    if scores is not None:
        result.update(age_group=AGE_GROUPS[int(scores.argmax())], age_scores=scores)
    return result


def age_words(group, arabic=False):
    """'about 25 to 32' / 'حوالي ٢٥ لـ ٣٢ سنة'."""
    if not group:
        return ""
    if group == "60+":
        return "فوق الستين" if arabic else "over 60"
    a, b = group.split("-")
    return f"عمره حوالي {a} لـ {b} سنة" if arabic else f"about {a} to {b} years old"


class Votes:
    """Steadies mood and age over several frames of the same person."""
    def __init__(self):
        self.age = np.zeros(len(AGE_GROUPS))
        self.moods = []

    def add(self, attrs):
        if "age_scores" in attrs:
            self.age += attrs["age_scores"]
        if "mood" in attrs:
            self.moods = (self.moods + [attrs["mood"]])[-5:]

    @property
    def age_group(self):
        return AGE_GROUPS[int(self.age.argmax())] if self.age.sum() else None

    @property
    def mood(self):
        return max(set(self.moods), key=self.moods.count) if self.moods else None

    @property
    def sentiment(self):
        m = self.mood
        return next((s for w, s in MOOD.values() if w == m), None) if m else None
