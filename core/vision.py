"""Geometry helpers: where is something (clock direction) and how far is it.

Clock-face directions are the standard way orientation & mobility trainers
describe positions to blind people: 12 o'clock is straight ahead, 1 o'clock
slightly right, 11 o'clock slightly left.

Distance uses the pinhole-camera model: distance = real_width * focal / width_px,
where the focal length in pixels comes from the camera's horizontal field of view.
"""
import math

HFOV_DEGREES = 70.0   # typical phone / webcam horizontal field of view

# Typical real-world width (metres) of each YOLO (COCO) class as seen by a camera.
REAL_WIDTH_M = {
    "person": 0.45, "bicycle": 1.7, "car": 1.8, "motorcycle": 0.8, "airplane": 30.0,
    "bus": 2.5, "train": 3.0, "truck": 2.5, "boat": 2.5, "traffic light": 0.35,
    "fire hydrant": 0.3, "stop sign": 0.75, "parking meter": 0.3, "bench": 1.5,
    "bird": 0.2, "cat": 0.4, "dog": 0.6, "horse": 1.6, "sheep": 1.0, "cow": 1.8,
    "elephant": 3.0, "bear": 1.5, "zebra": 2.0, "giraffe": 2.0, "backpack": 0.3,
    "umbrella": 1.0, "handbag": 0.3, "tie": 0.08, "suitcase": 0.45, "frisbee": 0.25,
    "skis": 1.6, "snowboard": 1.5, "sports ball": 0.22, "kite": 1.0,
    "baseball bat": 0.8, "baseball glove": 0.3, "skateboard": 0.8, "surfboard": 2.0,
    "tennis racket": 0.6, "bottle": 0.08, "wine glass": 0.08, "cup": 0.09,
    "fork": 0.18, "knife": 0.2, "spoon": 0.16, "bowl": 0.16, "banana": 0.18,
    "apple": 0.08, "sandwich": 0.12, "orange": 0.08, "broccoli": 0.15,
    "carrot": 0.18, "hot dog": 0.15, "pizza": 0.3, "donut": 0.1, "cake": 0.25,
    "chair": 0.5, "couch": 2.0, "potted plant": 0.35, "bed": 1.6,
    "dining table": 1.4, "toilet": 0.4, "tv": 1.0, "laptop": 0.34, "mouse": 0.07,
    "remote": 0.05, "keyboard": 0.45, "cell phone": 0.075, "microwave": 0.5,
    "oven": 0.6, "toaster": 0.3, "sink": 0.6, "refrigerator": 0.8, "book": 0.17,
    "clock": 0.3, "vase": 0.15, "scissors": 0.17, "teddy bear": 0.35,
    "hair drier": 0.25, "toothbrush": 0.02,
}
# Extra everyday objects (open-vocabulary model, models/glasses_vocab.pt)
EXTRA_WIDTH_M = {
    "door": 0.9,
    "stairs": 1.2,
    "escalator": 1.0,
    "elevator": 1.2,
    "window": 1.0,
    "wall": 3.0,
    "pole": 0.15,
    "crosswalk": 3.0,
    "trash can": 0.45,
    "wet floor sign": 0.3,
    "keys": 0.08,
    "glasses": 0.14,
    "sunglasses": 0.14,
    "wallet": 0.11,
    "pen": 0.14,
    "notebook": 0.2,
    "paper": 0.21,
    "medicine": 0.06,
    "pill bottle": 0.05,
    "pill box": 0.1,
    "charger": 0.06,
    "headphones": 0.18,
    "earphones": 0.03,
    "watch": 0.04,
    "ring": 0.02,
    "credit card": 0.085,
    "banknote": 0.15,
    "coin": 0.025,
    "plate": 0.25,
    "mug": 0.09,
    "glass of water": 0.08,
    "kettle": 0.2,
    "light switch": 0.08,
    "power outlet": 0.08,
    "door handle": 0.12,
    "faucet": 0.2,
    "towel": 0.5,
    "pillow": 0.5,
    "blanket": 1.5,
    "shoes": 0.3,
    "slippers": 0.25,
    "jacket": 0.5,
    "hat": 0.3,
    "bag": 0.35,
    "box": 0.4,
    "cabinet": 0.8,
    "shelf": 1.0,
    "mirror": 0.6,
    "comb": 0.15,
    "hairbrush": 0.2,
    "toothpaste": 0.18,
    "soap": 0.08,
    "shampoo bottle": 0.08,
    "tissue box": 0.22,
    "fire extinguisher": 0.18,
    "sign": 0.5,
    "fan": 0.45,
    "air conditioner": 0.9,
    "lamp": 0.3,
    "candle": 0.05,
    "calculator": 0.08,
    "tablet": 0.18,
    "cane": 0.03,
}
REAL_WIDTH_M.update({k: v for k, v in EXTRA_WIDTH_M.items() if k not in REAL_WIDTH_M})

FACE_WIDTH_M = 0.16   # face box width from the YuNet detector


def focal_px(image_width, hfov=HFOV_DEGREES):
    return (image_width / 2) / math.tan(math.radians(hfov / 2))


def estimate_distance(real_width_m, box_width_px, image_width):
    if not real_width_m or box_width_px <= 1:
        return None
    return real_width_m * focal_px(image_width) / box_width_px


def clock_direction(x1, x2, image_width, hfov=HFOV_DEGREES):
    """Return 11, 12 or 1 (o'clock) for the centre of a box."""
    centre = (x1 + x2) / 2 / image_width          # 0 = left edge, 1 = right edge
    angle = (centre - 0.5) * hfov                 # degrees, negative = left
    if angle < -12:
        return 11 if angle > -45 else 10
    if angle > 12:
        return 1 if angle < 45 else 2
    return 12


def rounded_distance(d):
    """Friendly distance value: None, 'close', or a number of metres."""
    if d is None:
        return None
    if d < 0.8:
        return "close"
    if d < 3:
        return round(d * 2) / 2      # nearest half metre
    if d > 15:
        return "far"
    return float(round(d))


def describe_box(label, x1, y1, x2, y2, image_width, image_height, real_width=None):
    """Structured info used by the speech builder and the on-screen boxes."""
    w = x2 - x1
    real = real_width if real_width is not None else REAL_WIDTH_M.get(label)
    dist = estimate_distance(real, w, image_width)
    return {
        "label": label,
        "box": [round(x1 / image_width, 4), round(y1 / image_height, 4),
                round(x2 / image_width, 4), round(y2 / image_height, 4)],
        "clock": clock_direction(x1, x2, image_width),
        "distance_m": round(dist, 2) if dist else None,
        "distance": rounded_distance(dist),
    }
