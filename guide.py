# ------------------------------
# Guide Mode - continuous, motion-aware assistance
# ------------------------------
# The page sends a frame every ~1.2 s. Instead of describing everything every
# time (exhausting for the user), Guide mode remembers the previous frames and
# speaks ONLY when something changes that matters:
#
#   1. Obstacle   - something big very close straight ahead      (highest priority)
#   2. Approaching - a person/object getting closer over time
#   3. Known face  - someone the user saved comes into view
#   4. Newcomer    - a person appears who wasn't there before
#
# Each kind of message has a cool-down so the same thing isn't repeated.
# ------------------------------
import threading
import time

import cv2

from core import i18n
from core.models import object_model
from detection_function import extra_objects, run_yolo
from face_function import analyse_faces

GUIDE_CONF = 0.40
OBSTACLE_DISTANCE = 1.0          # metres
APPROACH_RATIO = 0.75            # now <= 75% of the earlier distance -> approaching
APPROACH_MAX_DISTANCE = 4.0
TRACK_MEMORY = 4.0               # seconds a track is kept after it disappears
HAZARDS = {"stairs", "escalator", "pole", "wet floor sign", "door", "crosswalk", "trash can"}
HAZARD_DISTANCE = 3.0            # announce stairs / poles within this distance
OBSTACLE_LABELS = {"pole", "trash can", "wet floor sign", "person", "chair", "couch", "bench", "dining table", "bicycle", "motorcycle", "car",
                   "potted plant", "suitcase", "bed", "toilet", "tv", "refrigerator", "dog", "cat",
                   "backpack", "fire hydrant", "stop sign", "truck", "bus", "oven", "sink"}
COOLDOWN = {"hazard": 12.0, "obstacle": 5.0, "approach": 6.0, "known": 30.0, "newcomer": 15.0}
FRAME_WIDTH = 960                # frames are resized for speed


class Track:
    def __init__(self, det, now):
        self.label = det["label"]
        self.box = det["box"]
        self.history = [(now, det.get("distance_m"))]
        self.last_seen = now
        self.first_seen = now


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0


class GuideSession:
    def __init__(self):
        self.tracks = []
        self.last_said = {}        # key -> time
        self.lock = threading.Lock()

    def _cool(self, key, kind, now):
        if now - self.last_said.get(key, float("-inf")) < COOLDOWN[kind]:
            return False
        self.last_said[key] = now
        return True

    def _match(self, det, now):
        """Same object as before? Same label and overlapping or with a nearby centre
        (a box that grows quickly because someone walks closer barely overlaps)."""
        best, best_score = None, 0.0
        cx, cy = (det["box"][0] + det["box"][2]) / 2, (det["box"][1] + det["box"][3]) / 2
        for t in self.tracks:
            if t.label != det["label"] or t.last_seen == now:
                continue
            tx, ty = (t.box[0] + t.box[2]) / 2, (t.box[1] + t.box[3]) / 2
            closeness = 1 - min(1, ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5 / 0.2)
            score = max(_iou(t.box, det["box"]), closeness)
            if score > max(best_score, 0.25):
                best, best_score = t, score
        if best is None:
            best = Track(det, now)
            self.tracks.append(best)
            return best, True
        best.box = det["box"]
        best.last_seen = now
        best.history.append((now, det.get("distance_m")))
        best.history = [h for h in best.history if now - h[0] <= 3.5]
        return best, False

    def step(self, image, lang="en", now=None):
        """Analyse one frame. Returns (events, detections)."""
        now = time.time() if now is None else now
        h, w = image.shape[:2]
        if w > FRAME_WIDTH:
            image = cv2.resize(image, (FRAME_WIDTH, int(h * FRAME_WIDTH / w)))
        dets = run_yolo(image, object_model(), GUIDE_CONF) + extra_objects(image, HAZARDS, 0.45)
        for d in dets:
            d["kind"] = "object"
            d["name"] = i18n.object_name(d["label"], 1, lang)

        people = []
        if any(d["label"] == "person" for d in dets):
            ih, iw = image.shape[:2]
            boxes = [[d["box"][0] * iw, d["box"][1] * ih, d["box"][2] * iw, d["box"][3] * ih]
                     for d in dets if d["label"] == "person"]
            people = analyse_faces(image, boxes=boxes, lang=lang)

        events = []
        with self.lock:
            for d in dets:
                track, is_new = self._match(d, now)
                dist = d.get("distance_m")
                label = d["label"]
                name = i18n.object_name(label, 1, lang)
                place = i18n.where(d, lang)

                # 1. obstacle straight ahead
                limit = 0.6 if label == "person" else OBSTACLE_DISTANCE   # people in a conversation stand close
                if (label in OBSTACLE_LABELS and dist is not None and dist <= limit
                        and d["clock"] == 12 and self._cool(f"obstacle:{label}", "obstacle", now)):
                    text = (f"انتبه! {name} قريب جدًا قدامك." if i18n.is_ar(lang)
                            else f"Careful! {name.capitalize()} very close in front of you.")
                    events.append({"type": "obstacle", "priority": 1, "text": text})
                    continue

                # 1b. hazards: stairs, escalator, crossing... announce once when they appear
                if (label in HAZARDS and is_new and (dist is None or dist <= HAZARD_DISTANCE)
                        and self._cool(f"hazard:{label}", "hazard", now)):
                    text = (f"انتبه، فيه {name} {place}." if i18n.is_ar(lang)
                            else f"Heads up: {i18n.counted(label, 1, lang)} {place}.")
                    events.append({"type": "hazard", "priority": 1, "text": text})
                    continue

                # 2. approaching: distance shrinking over the last few seconds
                old = [hd for ht, hd in track.history if hd and now - ht >= 1.0]
                if (not is_new and dist and old and dist <= APPROACH_MAX_DISTANCE
                        and dist <= old[0] * APPROACH_RATIO and self._cool(f"approach:{id(track)}", "approach", now)):
                    text = (f"{name} بيقرب منك {place}." if i18n.is_ar(lang)
                            else f"{name.capitalize()} approaching {place}.")
                    events.append({"type": "approach", "priority": 2, "text": text})
                    continue

                # 4. a new person walks into view (if not identified below)
                if label == "person" and is_new and self._cool("newcomer", "newcomer", now):
                    text = (f"فيه حد {place}." if i18n.is_ar(lang) else f"Someone is {place}.")
                    events.append({"type": "newcomer", "priority": 4, "text": text})

            # 3. known faces
            for p in people:
                if p.get("certainty") == "sure" and self._cool(f"known:{p['person']}", "known", now):
                    text = (f"{p['person']} {i18n.where(p, lang)}." if i18n.is_ar(lang)
                            else f"{p['person']} is {i18n.where(p, lang)}.")
                    events.append({"type": "known", "priority": 3, "text": text})

            self.tracks = [t for t in self.tracks if now - t.last_seen <= TRACK_MEMORY]

        # a known face replaces the generic "someone is ..." message
        if any(e["type"] == "known" for e in events):
            events = [e for e in events if e["type"] != "newcomer"]
        events.sort(key=lambda e: e["priority"])
        return events[:2], dets + people


_sessions = {}
_sessions_lock = threading.Lock()


def get_session(session_id):
    with _sessions_lock:
        if session_id not in _sessions:
            _sessions[session_id] = GuideSession()
        return _sessions[session_id]


def reset_session(session_id):
    with _sessions_lock:
        _sessions.pop(session_id, None)
