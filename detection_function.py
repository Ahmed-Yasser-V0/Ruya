# ------------------------------
# Object & Currency Detection
# ------------------------------
# Uses the shared YOLO models (loaded once in core/models.py).
# Returns a spoken sentence AND structured boxes for the on-screen overlay.
#
#   "Two people: the nearest at 12 o'clock, about 1 meter away. A chair at 11 o'clock, ..."
#   "A 100 pound note and a 20 pound note. Total: 120 pounds."
# ------------------------------
import cv2

from core import i18n
from core.models import INFERENCE_LOCK, currency_model, object_model, open_vocab_model
from core.vision import describe_box

OBJECT_CONF = 0.35     # below this YOLO guesses too often
CURRENCY_CONF = 0.50   # low-confidence "notes" were often phones or pets
MAX_GROUPS = 4         # keep speech short: the 4 nearest kinds of things
EXTRA_CONF = 0.40      # open-vocabulary objects need more certainty
# Everyday things worth mentioning in "detect" that YOLO's 80 classes don't have
EXTRA_LABELS = {"door", "stairs", "escalator", "elevator", "keys", "glasses", "sunglasses", "wallet",
                "medicine", "pill bottle", "charger", "headphones", "power outlet", "light switch",
                "trash can", "pole", "fire extinguisher", "shoes", "cane", "wet floor sign", "crosswalk"}


def extra_objects(image, labels=EXTRA_LABELS, conf=EXTRA_CONF):
    model = open_vocab_model()
    if model is None:
        return []
    return [d for d in run_yolo(image, model, conf) if d["label"] in labels]


def _load(frame):
    return cv2.imread(frame) if isinstance(frame, str) else frame


def run_yolo(image, model, conf):
    """Run YOLO and return a list of described boxes, nearest first."""
    h, w = image.shape[:2]
    with INFERENCE_LOCK:
        results = model.predict(image, conf=conf, verbose=False)
    found = []
    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls)]
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            info = describe_box(label, x1, y1, x2, y2, w, h)
            info["confidence"] = round(float(box.conf), 2)
            found.append(info)
    found.sort(key=lambda d: d["distance_m"] if d["distance_m"] else 99)
    return found


def objects_sentence(found, lang):
    if not found:
        return i18n.msg("nothing_detected", lang)
    groups = {}
    for d in found:                       # already nearest-first
        groups.setdefault(d["label"], []).append(d)
    parts = []
    for label, items in list(groups.items())[:MAX_GROUPS]:
        what = i18n.counted(label, len(items), lang)
        place = i18n.where(items[0], lang)
        if len(items) > 1:
            nearest = "أقربهم" if i18n.is_ar(lang) else "the nearest"
            parts.append(f"{what}: {nearest} {place}" if place else what)
        else:
            parts.append(f"{what} {place}" if place else what)
    return " ".join(i18n.sentence(p) for p in parts)


def currency_sentence(found, lang):
    if not found:
        return i18n.msg("no_currency", lang)
    counts = {}
    for d in found:
        counts[d["label"]] = counts.get(d["label"], 0) + 1
    names, total = [], 0
    for label, n in counts.items():
        en, ar = i18n.CURRENCY.get(label, (label, label))
        name = ar if i18n.is_ar(lang) else en
        total += i18n.CURRENCY_VALUE.get(label, 0) * n
        if n == 1:
            names.append(name if i18n.is_ar(lang) else ("an " if name[0] in "aeiou" else "a ") + name)
        else:
            names.append(f"{i18n.num(n, lang)} {name}" if i18n.is_ar(lang) else f"{i18n.num(n, lang)} {name}s")
    text = i18n.sentence(i18n.join(names, lang))
    if len(found) > 1:
        t = int(total) if float(total).is_integer() else total
        text += f" المجموع {t} جنيه." if i18n.is_ar(lang) else f" Total: {t} pounds."
    return text


def detect(frame, mode="object", lang="en"):
    """Returns (spoken_text, detections)."""
    image = _load(frame)
    if image is None:
        return i18n.msg("no_image", lang), []
    if mode == "currency":
        found = run_yolo(image, currency_model(), CURRENCY_CONF)
        for d in found:
            d["kind"] = "currency"
            d["name"] = i18n.CURRENCY.get(d["label"], (d["label"],) * 2)[1 if i18n.is_ar(lang) else 0]
        return currency_sentence(found, lang), found
    found = run_yolo(image, object_model(), OBJECT_CONF) + extra_objects(image)
    found.sort(key=lambda d: d["distance_m"] if d["distance_m"] else 99)
    for d in found:
        d["kind"] = "object"
        d["name"] = i18n.object_name(d["label"], 1, lang)
    return objects_sentence(found, lang), found


# --- kept for backwards compatibility with the original project ---
def image_detection(frame, mode, lang="en"):
    return detect(frame, mode, lang)[0]


def test_detection():
    detect("static/logo.png", "object")
    find_model = open_vocab_model()
    print("Open vocabulary:", "ready" if find_model is not None else "not installed")
    detect("static/logo.png", "currency")
