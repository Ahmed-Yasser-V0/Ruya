# ------------------------------
# Find Object + Distance
# ------------------------------
# "find bottle" -> "The bottle is at 1 o'clock, about 1 meter away."
#
# Fixes vs. the original version:
#  * The size table used old class names (tvmonitor, sofa, pottedplant...) that
#    YOLOv8 doesn't use, so finding a TV / couch / plant / table crashed.
#  * Words are normalised: "Bottle", "bottles", "the bottle", "a phone",
#    Arabic ("إزازة", "موبايل") all work.
#  * The most confident match is used (not just the first box).
#  * Distance is calibrated from the camera's field of view and given in metres.
# ------------------------------
import re

from core import i18n
from core.models import object_model, open_vocab_model
from detection_function import _load, run_yolo

FIND_CONF = 0.30

# Extra words people say -> YOLO class name
SYNONYMS = {
    "phone": "cell phone", "mobile": "cell phone", "mobile phone": "cell phone", "iphone": "cell phone",
    "smartphone": "cell phone", "telephone": "cell phone", "sofa": "couch", "television": "tv",
    "tv monitor": "tv", "screen": "tv", "monitor": "tv", "plant": "potted plant", "flower": "potted plant",
    "table": "dining table", "desk": "dining table", "bike": "bicycle", "motorbike": "motorcycle",
    "aeroplane": "airplane", "plane": "airplane", "glass": "wine glass", "mug": "cup", "bag": "handbag",
    "purse": "handbag", "ball": "sports ball", "man": "person", "woman": "person", "someone": "person",
    "somebody": "person", "people": "person", "kid": "person", "child": "person", "fridge": "refrigerator",
    "computer": "laptop", "notebook": "laptop", "kitten": "cat", "puppy": "dog", "doughnut": "donut",
    "hairdryer": "hair drier", "hair dryer": "hair drier", "controller": "remote", "suitcase": "suitcase",
    "luggage": "suitcase", "water": "bottle", "water bottle": "bottle",
    # Arabic (Egyptian + standard)
    "موبايل": "cell phone", "تليفون": "cell phone", "فون": "cell phone", "هاتف": "cell phone", "جوال": "cell phone",
    "ازازة": "bottle", "ازازه": "bottle", "إزازة": "bottle", "قزازة": "bottle", "زجاجة": "bottle", "زجاجه": "bottle", "مياه": "bottle",
    "كوباية": "cup", "كوبايه": "cup", "كوب": "cup", "مج": "cup", "فنجان": "cup",
    "كرسي": "chair", "كرسى": "chair", "كنبة": "couch", "كنبه": "couch", "ترابيزة": "dining table", "ترابيزه": "dining table",
    "طاولة": "dining table", "مكتب": "dining table", "لابتوب": "laptop", "لاب": "laptop", "كمبيوتر": "laptop",
    "شنطة": "handbag", "شنطه": "handbag", "شنطة ضهر": "backpack", "كتاب": "book", "ساعة": "clock", "ساعه": "clock",
    "تلفزيون": "tv", "شاشة": "tv", "شاشه": "tv", "ريموت": "remote", "كيبورد": "keyboard", "ماوس": "mouse",
    "مقص": "scissors", "شخص": "person", "حد": "person", "راجل": "person", "ست": "person", "عربية": "car", "عربيه": "car",
    "سيارة": "car", "عجلة": "bicycle", "قطة": "cat", "قطه": "cat", "كلب": "dog", "زرعة": "potted plant", "زرعه": "potted plant",
    "سرير": "bed", "تلاجة": "refrigerator", "تلاجه": "refrigerator", "شمسية": "umbrella", "كورة": "sports ball", "كوره": "sports ball",
    "طبق": "bowl", "معلقة": "spoon", "معلقه": "spoon", "شوكة": "fork", "سكينة": "knife", "سكينه": "knife",
    "تفاحة": "apple", "موزة": "banana", "برتقانة": "orange", "فرشة": "toothbrush", "سشوار": "hair drier",
}
# Words for the open-vocabulary model (things YOLO's 80 classes don't have)
OPEN_SYNONYMS = {
    "key": "keys", "my keys": "keys", "eyeglasses": "glasses", "spectacles": "glasses", "shades": "sunglasses",
    "purse wallet": "wallet", "money": "banknote", "cash": "banknote", "pills": "medicine", "tablets": "medicine",
    "medication": "medicine", "staircase": "stairs", "steps": "stairs", "stair": "stairs", "lift": "elevator",
    "socket": "power outlet", "plug": "power outlet", "outlet": "power outlet", "switch": "light switch",
    "handle": "door handle", "tap": "faucet", "bin": "trash can", "garbage": "trash can", "rubbish bin": "trash can",
    "shoe": "shoes", "slipper": "slippers", "headset": "headphones", "earbuds": "earphones", "card": "credit card",
    "tissues": "tissue box", "walking stick": "cane", "stick": "cane",
    "مفتاح": "keys", "مفاتيح": "keys", "المفاتيح": "keys", "نضارة": "glasses", "نضاره": "glasses", "نظارة": "glasses",
    "نظاره": "glasses", "محفظة": "wallet", "محفظه": "wallet", "قلم": "pen", "دوا": "medicine", "دواء": "medicine",
    "برشام": "medicine", "علبة الدوا": "pill bottle", "شاحن": "charger", "سماعة": "headphones", "سماعه": "headphones",
    "باب": "door", "الباب": "door", "سلم": "stairs", "السلم": "stairs", "سلالم": "stairs", "أسانسير": "elevator",
    "اسانسير": "elevator", "شباك": "window", "فيشة": "power outlet", "فيشه": "power outlet", "مفتاح النور": "light switch",
    "اوكرة": "door handle", "أوكرة": "door handle", "حنفية": "faucet", "فوطة": "towel", "مخدة": "pillow",
    "جزمة": "shoes", "جزمه": "shoes", "شبشب": "slippers", "طاقية": "hat", "كاب": "hat", "مراية": "mirror", "مرايه": "mirror",
    "مشط": "comb", "صابون": "soap", "مناديل": "tissue box", "طفاية": "fire extinguisher", "مروحة": "fan", "كارت": "credit card",
    "سلة": "trash can", "زبالة": "trash can", "ساعة ايد": "watch", "خاتم": "ring", "عصاية": "cane", "كاتل": "kettle",
}
OPEN_CONF = 0.30

STOP_WORDS = {"the", "a", "an", "my", "me", "for", "find", "where", "is", "are", "please", "look",
              "عن", "على", "فين", "ال", "دور", "ابحث", "لي", "ليا"}


def normalise(query):
    """Turn what the user said/typed into a YOLO class name (or None).
    Returns names from YOLO's 80 classes first, then from the open vocabulary."""
    if not query:
        return None
    names = set(object_model().names.values())
    ov = open_vocab_model()
    open_names = set(ov.names.values()) if ov is not None else set()
    q = re.sub(r"[^\w\s]", " ", str(query).lower(), flags=re.UNICODE)
    words = [w for w in q.split() if w not in STOP_WORDS]
    if not words:
        return None
    candidates = [" ".join(words)] + [" ".join(words[i:i + 2]) for i in range(len(words) - 1)] + words
    for c in candidates:
        for form in (c, c[2:] if c.startswith("ال") else c, c.rstrip("s"), c[:-2] if c.endswith("es") else c):
            if form in names:
                return form
            if form in SYNONYMS:
                return SYNONYMS[form]
            if form in OPEN_SYNONYMS and OPEN_SYNONYMS[form] in open_names:
                return OPEN_SYNONYMS[form]
            if form in open_names:
                return form
            if form + "s" in open_names:          # "key" -> "keys"
                return form + "s"
    return None


def find_object(frame, query, lang="en"):
    """Returns (spoken_text, detections)."""
    if not query or not str(query).strip():
        return i18n.msg("what_to_find", lang), []
    target = normalise(query)
    if target is None:
        if i18n.is_ar(lang):
            return f"معرفش أدور على \"{query}\". أقدر ألاقي حاجات زي الكرسي والإزازة والموبايل.", []
        return f"I can't look for \"{query}\" yet. I can find things like a chair, a bottle or a phone.", []

    image = _load(frame)
    if image is None:
        return i18n.msg("no_image", lang), []
    coco = set(object_model().names.values())
    if target in coco:
        matches = [d for d in run_yolo(image, object_model(), FIND_CONF) if d["label"] == target]
    else:
        matches = [d for d in run_yolo(image, open_vocab_model(), OPEN_CONF) if d["label"] == target]
    name = i18n.object_name(target, 1, lang)
    if not matches:
        if i18n.is_ar(lang):
            return f"مش لاقي {name}.", []
        return f"I can't find the {name}.", []

    best = max(matches, key=lambda d: d["confidence"])
    best["kind"] = "object"
    best["name"] = name
    if i18n.is_ar(lang):
        text = f"لقيت {name} {i18n.where(best, lang)}."
        if len(matches) > 1:
            text += f" وفيه {i18n.num(len(matches), lang)} في الصورة."
    else:
        text = f"The {name} is {i18n.where(best, lang)}."
        if len(matches) > 1:
            text += f" I see {i18n.num(len(matches), lang)} of them; this is the clearest one."
    return text, [best]


# --- kept for backwards compatibility with the original project ---
def find(image_path, object_to_be_found, lang="en"):
    return find_object(image_path, object_to_be_found, lang)[0]


def test_find():
    find_object("static/logo.png", "cat")
