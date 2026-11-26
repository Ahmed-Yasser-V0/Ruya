# ------------------------------
# Face Recognition Mode
# ------------------------------
# Recognises people the user has added (family, friends) and says who is in
# front of them, where (clock direction) and how far:
#     "Ahmed is at 12 o'clock, about 1 meter away."
#
# Models (OpenCV Zoo, Apache-2.0 / MIT, run with OpenCV - no extra libraries):
#   - YuNet : face detector (box + 5 landmarks)
#   - SFace : face recogniser (128-number "embedding" per face)
#
# Storage (unchanged, so faces saved with the earlier version still work):
#   faces/<Name>/<photo>.jpg   enrollment photos (3 per person is best)
#   faces/face_db.npz          cached embeddings, rebuilt when photos change
#
# Decision rule (tuned on 65 real faces from the user's own camera):
#   score >= 0.42 and at least 0.08 ahead of the next person  -> name
#   score >= 0.36 and at least 0.05 ahead                     -> "might be <name>"
#   otherwise                                                 -> unknown person
# Compared with a single 0.363 threshold this removed every wrong-name answer
# (2 -> 0) on that test set while keeping 87% of faces named.
#
# Privacy: only people added on purpose can be named; nothing about unknown
# people is stored.
# ------------------------------
import os
import re
import shutil
from datetime import datetime

import cv2
import numpy as np

from core import face_attributes, i18n
from core.models import INFERENCE_LOCK, path
from core.vision import FACE_WIDTH_M, describe_box

# --- PATHS & SETTINGS ---
MODELS_DIR = path("models")
DETECTOR_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
RECOGNIZER_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")
FACES_DIR = path("faces")
DB_PATH = os.path.join(FACES_DIR, "face_db.npz")

DETECTION_SCORE = 0.7        # 0.8 missed faces that were turned or half in frame
NAME_THRESHOLD = 0.42        # confident name
MAYBE_THRESHOLD = 0.36       # "might be ..."
NAME_MARGIN = 0.08           # best person must beat the second-best by this much
MAYBE_MARGIN = 0.05
MATCH_THRESHOLD = NAME_THRESHOLD   # kept for backwards compatibility
MIN_FACE_TO_RECOGNISE = 40   # px; smaller faces are too blurry to identify
MIN_FACE_FOR_MOOD = 40       # px; expression needs a reasonably clear face
MIN_FACE_FOR_AGE = 60        # px; age guesses on smaller faces were unreliable in testing
MIN_FACE_TO_ENROLL = 80      # px; enrollment photos must be clear
MIN_SHARPNESS = 50           # Laplacian variance of the aligned face
MIN_BRIGHTNESS = 35          # mean grey level of the aligned face
MAX_IMAGE_SIDE = 1280
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# --- MODELS ---
face_detector = cv2.FaceDetectorYN.create(DETECTOR_PATH, "", (320, 320), DETECTION_SCORE, 0.3, 5000)
face_recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_PATH, "")


# ------------------------------
# Helpers
# ------------------------------
def load_image(frame):
    image = cv2.imread(frame) if isinstance(frame, str) else frame
    if image is None:
        return None
    h, w = image.shape[:2]
    scale = MAX_IMAGE_SIDE / max(h, w)
    if scale < 1:
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    return image


def detect_faces(image):
    """YuNet rows [x, y, w, h, 10 landmark values, score], left to right."""
    h, w = image.shape[:2]
    with INFERENCE_LOCK:
        face_detector.setInputSize((w, h))
        _, faces = face_detector.detect(image)
    if faces is None:
        return []
    return sorted(faces, key=lambda f: f[0])


def aligned(image, face):
    with INFERENCE_LOCK:
        return face_recognizer.alignCrop(image, face)


def get_embedding(image, face):
    crop = aligned(image, face)
    with INFERENCE_LOCK:
        feature = face_recognizer.feature(crop).flatten()
    return feature / np.linalg.norm(feature)


def face_quality(image, face):
    """Returns None if the face is good enough to save, else a reason key."""
    if face[2] < MIN_FACE_TO_ENROLL:
        return "far"
    gray = cv2.cvtColor(aligned(image, face), cv2.COLOR_BGR2GRAY)
    if gray.mean() < MIN_BRIGHTNESS:
        return "dark"
    if cv2.Laplacian(gray, cv2.CV_64F).var() < MIN_SHARPNESS:
        return "blurry"
    lm = face[4:14].reshape(5, 2)          # right eye, left eye, nose, mouth corners
    eye_span = max(1.0, lm[1][0] - lm[0][0])
    yaw = (lm[2][0] - lm[0][0]) / eye_span
    if yaw < 0.15 or yaw > 0.85:
        return "turned"
    return None


def clean_name(name):
    name = re.sub(r"[^\w\s-]", "", name or "", flags=re.UNICODE).strip()
    return " ".join(part[:1].upper() + part[1:] for part in name.split())


def person_folder(name):
    return os.path.join(FACES_DIR, name.replace(" ", "_"))


def folder_to_name(folder):
    return folder.replace("_", " ")


def person_boxes(image):
    """Person boxes from the shared YOLO model (class 0 = person)."""
    from core.models import object_model
    with INFERENCE_LOCK:
        results = object_model().predict(image, classes=[0], conf=0.3, verbose=False)
    return [box.xyxy[0].tolist() for r in results for box in r.boxes]


def keep_real_faces(image, faces, boxes=None):
    """Drop 'faces' that are not on a person (YuNet sometimes fires on pet faces)."""
    if len(faces) == 0:
        return faces
    boxes = person_boxes(image) if boxes is None else boxes
    kept = []
    for face in faces:
        cx, cy = face[0] + face[2] / 2, face[1] + face[3] / 2
        if any(x1 <= cx <= x2 and y1 <= cy <= y2 for x1, y1, x2, y2 in boxes):
            kept.append(face)
    return kept


# ------------------------------
# Face database
# ------------------------------
def _photos():
    items = []
    if os.path.isdir(FACES_DIR):
        for person in sorted(os.listdir(FACES_DIR)):
            folder = os.path.join(FACES_DIR, person)
            if os.path.isdir(folder):
                for photo in sorted(os.listdir(folder)):
                    if photo.lower().endswith(IMAGE_EXTENSIONS):
                        items.append((person, os.path.join(folder, photo)))
    return items


def _signature():
    return "|".join(f"{p}:{os.path.getmtime(p)}" for _, p in _photos())


def build_face_db():
    names, embeddings = [], []
    for person, photo in _photos():
        image = load_image(photo)
        if image is None:
            continue
        faces = detect_faces(image)
        if not faces:
            print(f"[face] No face found in {photo}, skipped")
            continue
        face = max(faces, key=lambda f: f[2] * f[3])
        names.append(folder_to_name(person))
        embeddings.append(get_embedding(image, face))
    os.makedirs(FACES_DIR, exist_ok=True)
    matrix = np.array(embeddings).reshape(-1, 128)
    np.savez(DB_PATH, names=np.array(names), embeddings=matrix, signature=np.array(_signature()))
    return names, matrix


def load_face_db():
    signature = _signature()
    if os.path.exists(DB_PATH):
        try:
            data = np.load(DB_PATH)
            if str(data["signature"]) == signature:
                return list(data["names"]), data["embeddings"]
        except Exception:
            pass
    return build_face_db()


def rank_people(embedding, names, embeddings):
    """[(name, best_score), ...] best first (one entry per person)."""
    if len(names) == 0:
        return []
    sims = embeddings @ embedding
    best = {}
    for name, s in zip(names, sims):
        best[name] = max(best.get(name, -1.0), float(s))
    return sorted(best.items(), key=lambda kv: -kv[1])


def decide(ranked):
    """Returns (name or None, score, certainty) with certainty in {'sure','maybe','unknown'}."""
    if not ranked:
        return None, 0.0, "unknown"
    name, best = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else -1.0
    if best >= NAME_THRESHOLD and best - second >= NAME_MARGIN:
        return name, best, "sure"
    if best >= MAYBE_THRESHOLD and best - second >= MAYBE_MARGIN:
        return name, best, "maybe"
    return None, best, "unknown"


def match_face(embedding, names, embeddings):
    """Backwards-compatible: (name or None, score). Only confident names."""
    name, score, certainty = decide(rank_people(embedding, names, embeddings))
    return (name if certainty == "sure" else None), score


# ------------------------------
# Recognition
# ------------------------------
def analyse_faces(image, boxes=None, lang="en"):
    """Structured result for every face: name, certainty, clock, distance, box."""
    h, w = image.shape[:2]
    faces = keep_real_faces(image, detect_faces(image), boxes)
    names, embeddings = load_face_db()
    people = []
    for face in faces:
        x, y, fw, fh = [float(v) for v in face[:4]]
        info = describe_box("face", x, y, x + fw, y + fh, w, h, FACE_WIDTH_M)
        info["kind"] = "face"
        if fw < MIN_FACE_TO_RECOGNISE:
            info.update(person=None, certainty="far", score=0.0)
        else:
            person, score, certainty = decide(rank_people(get_embedding(image, face), names, embeddings))
            info.update(person=person, certainty=certainty, score=round(score, 3))
        if fw >= MIN_FACE_FOR_MOOD:
            attrs = face_attributes.analyse(image, face)
            if attrs.get("mood"):
                info.update(mood=attrs["mood"], sentiment=attrs["sentiment"])
            if fw >= MIN_FACE_FOR_AGE and attrs.get("age_group"):
                info["age_group"] = attrs["age_group"]
        if info["person"] and info["certainty"] == "sure":
            info["name"] = info["person"]
        elif info["person"]:
            info["name"] = ("ممكن " if i18n.is_ar(lang) else "Maybe ") + info["person"]
        else:
            info["name"] = "مش معروف" if i18n.is_ar(lang) else "Unknown"
        people.append(info)
    return people


def mood_words(p, ar):
    """', and looks happy' - only when the face shows something (calm is not announced)."""
    m = p.get("mood")
    if not m or m == "calm":
        return ""
    return f"، وشكله {face_attributes.MOOD_AR.get(m, m)}" if ar else f", and looks {m}"


def faces_sentence(people, lang="en"):
    if not people:
        return i18n.msg("no_faces", lang)
    ar = i18n.is_ar(lang)
    parts, unknown, far = [], [], 0
    for p in people:
        place = i18n.where(p, lang)
        if p["certainty"] == "sure":
            parts.append((f"{p['person']} {place}" if ar else f"{p['person']} is {place}") + mood_words(p, ar))
        elif p["certainty"] == "maybe":
            parts.append((f"غالبًا {p['person']} {place}" if ar else f"this might be {p['person']}, {place}")
                         + mood_words(p, ar))
        elif p["certainty"] == "far":
            far += 1
        else:
            unknown.append(p)
    if len(unknown) == 1:
        u = unknown[0]
        age = face_attributes.age_words(u.get("age_group"), ar)
        if ar:
            parts.append(f"شخص مش معروف {i18n.where(u, lang)}" + (f"، {age}" if age else "") + mood_words(u, ar))
        else:
            parts.append(f"an unknown person{', ' + age + ',' if age else ''} is {i18n.where(u, lang)}"
                         + mood_words(u, ar))
    elif len(unknown) > 1:
        group = f"{i18n.num(len(unknown), lang)} أشخاص مش معروفين" if ar else f"{i18n.num(len(unknown), lang)} unknown people"
        moods = [u.get("mood") for u in unknown if u.get("mood")]
        top = max(set(moods), key=moods.count) if moods else None
        if top and top != "calm" and moods.count(top) * 2 > len(unknown):     # say it only if most share it
            group += f"، وأغلبهم {face_attributes.MOOD_AR.get(top, top)}" if ar else f", most of them look {top}"
        parts.append(group)
    text = i18n.sentence(i18n.join(parts, lang)) if parts else ""
    if far:
        extra = (f"وفيه {i18n.num(far, lang)} بعيد عشان أعرفه." if ar else
                 f"{i18n.num(far, lang)} {'person is' if far == 1 else 'people are'} too far away to recognise.")
        text = (text + " " + i18n.sentence(extra)).strip()
    return text


def recognize(frame, lang="en"):
    """Returns (spoken_text, detections)."""
    image = load_image(frame)
    if image is None:
        return i18n.msg("no_image", lang), []
    people = analyse_faces(image, lang=lang)
    return faces_sentence(people, lang), people


# ------------------------------
# Enrollment
# ------------------------------
QUALITY_TEXT = {
    "far": ("{name} is too far away. Move closer.", "{name} بعيد. قرّب شوية."),
    "dark": ("It's too dark to see {name}'s face clearly.", "الإضاءة ضعيفة ومش شايف وش {name} كويس."),
    "blurry": ("The photo is blurry. Hold still.", "الصورة مهزوزة. اثبت شوية."),
    "turned": ("Please face the camera a bit more.", "بص للكاميرا أكتر شوية."),
}


def enroll(frames, name, lang="en"):
    """Save one or more photos of a person (guided enrollment sends 3).

    Returns (spoken_text, details) where details has saved / rejected counts.
    """
    ar = i18n.is_ar(lang)
    name = clean_name(name)
    if not name:
        return i18n.msg("need_name", lang), {"saved": 0}
    names, embeddings = load_face_db()
    saved, reasons, also_like = 0, [], None
    folder = person_folder(name)
    for frame in frames:
        image = load_image(frame)
        if image is None:
            continue
        faces = keep_real_faces(image, detect_faces(image))
        if not faces:
            reasons.append("none")
            continue
        if len(faces) > 1:
            reasons.append("many")
            continue
        problem = face_quality(image, faces[0])
        if problem:
            reasons.append(problem)
            continue
        ranked = rank_people(get_embedding(image, faces[0]), names, embeddings)
        other, _, certainty = decide(ranked)
        if other and other != name and certainty == "sure":
            also_like = other
        os.makedirs(folder, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        cv2.imwrite(os.path.join(folder, f"{stamp}.jpg"), image)
        saved += 1

    if saved:
        build_face_db()
    total = len([p for p in os.listdir(folder) if p.lower().endswith(IMAGE_EXTENSIONS)]) if os.path.isdir(folder) else 0

    if not saved:
        r = reasons[0] if reasons else "none"
        if r == "none":
            text = (f"مش شايف وش. وجّه الكاميرا ناحية {name} وجرب تاني." if ar
                    else f"I couldn't see a face. Point the camera at {name} and try again.")
        elif r == "many":
            text = (f"شايف أكتر من وش. خلي {name} لوحده قدام الكاميرا." if ar
                    else f"I can see more than one face. Make sure only {name} is in front of the camera.")
        else:
            text = QUALITY_TEXT[r][1 if ar else 0].format(name=name)
        return text, {"saved": 0, "total": total, "rejected": reasons}

    if ar:
        photos = {1: "صورة واحدة", 2: "صورتين"}.get(total, f"{i18n.num(total, lang)} صور")
        text = f"تمام، حفظت {name}. عندي {photos} ليه."
        if total < 3:
            text += " ضيف صور كمان من زوايا مختلفة عشان أعرفه أحسن."
        if also_like:
            text += f" ملحوظة: الوش ده شبه {also_like}."
    else:
        text = f"Saved {name}. I now have {i18n.num(total, lang)} photo{'s' if total != 1 else ''} of {name}."
        if total < 3:
            text += " Add a couple more from different angles for better recognition."
        if also_like:
            text += f" Note: this face also looks like {also_like}."
    return text, {"saved": saved, "total": total, "rejected": reasons}


def forget(name, lang="en"):
    name = clean_name(name)
    folder = person_folder(name)
    if not name or not os.path.isdir(folder):
        return (f"معرفش حد اسمه {name or 'كده'}." if i18n.is_ar(lang)
                else f"I don't know anyone called {name or 'that'}.")
    shutil.rmtree(folder)
    build_face_db()
    return f"نسيت {name}." if i18n.is_ar(lang) else f"I have forgotten {name}."


def known_people_sentence(lang="en"):
    names, _ = load_face_db()
    people = sorted(set(names))
    if not people:
        return i18n.msg("known_none", lang)
    joined = i18n.join(people, lang)
    return f"أعرف {joined}." if i18n.is_ar(lang) else f"I know {joined}."


# --- kept for backwards compatibility with the original API ---
def recognize_faces(frame, lang="en"):
    return recognize(frame, lang)[0]


def add_face(frame, name, lang="en"):
    return enroll([frame], name, lang)[0]


def forget_face(name, lang="en"):
    return forget(name, lang)


def list_known_people(lang="en"):
    return known_people_sentence(lang)


def test_face():
    load_face_db()
    recognize("images/people.jpg")
