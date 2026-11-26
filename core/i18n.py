"""English / Arabic sentences built inside the app (no online translation needed).

Only free text from the AI (scene captions, text read from paper) still needs
translation; everything the app says itself is written here in both languages.
"""

# ---------- Object names (YOLO / COCO classes) ----------
AR_NAMES = {
    "person": ("شخص", "أشخاص"), "bicycle": ("عجلة", "عجل"), "car": ("عربية", "عربيات"),
    "motorcycle": ("موتوسيكل", "موتوسيكلات"), "airplane": ("طيارة", "طيارات"),
    "bus": ("أتوبيس", "أتوبيسات"), "train": ("قطر", "قطارات"), "truck": ("نقل", "عربيات نقل"),
    "boat": ("مركب", "مراكب"), "traffic light": ("إشارة مرور", "إشارات مرور"),
    "fire hydrant": ("حنفية حريق", "حنفيات حريق"), "stop sign": ("علامة قف", "علامات قف"),
    "parking meter": ("عداد ركن", "عدادات ركن"), "bench": ("دكة", "دكك"),
    "bird": ("عصفورة", "عصافير"), "cat": ("قطة", "قطط"), "dog": ("كلب", "كلاب"),
    "horse": ("حصان", "أحصنة"), "sheep": ("خروف", "خرفان"), "cow": ("بقرة", "بقر"),
    "elephant": ("فيل", "أفيال"), "bear": ("دب", "دببة"), "zebra": ("حمار وحشي", "حمير وحشية"),
    "giraffe": ("زرافة", "زرافات"), "backpack": ("شنطة ضهر", "شنط ضهر"),
    "umbrella": ("شمسية", "شماسي"), "handbag": ("شنطة", "شنط"), "tie": ("كرافتة", "كرافتات"),
    "suitcase": ("شنطة سفر", "شنط سفر"), "frisbee": ("فريسبي", "فريسبي"),
    "skis": ("زلاجات", "زلاجات"), "snowboard": ("لوح تزلج", "ألواح تزلج"),
    "sports ball": ("كورة", "كور"), "kite": ("طيارة ورق", "طيارات ورق"),
    "baseball bat": ("مضرب بيسبول", "مضارب بيسبول"), "baseball glove": ("جوانتي بيسبول", "جوانتيات"),
    "skateboard": ("سكيت بورد", "سكيت بورد"), "surfboard": ("لوح ركمجة", "ألواح ركمجة"),
    "tennis racket": ("مضرب تنس", "مضارب تنس"), "bottle": ("إزازة", "إزايز"),
    "wine glass": ("كاس", "كاسات"), "cup": ("كوباية", "كوبايات"), "fork": ("شوكة", "شوك"),
    "knife": ("سكينة", "سكاكين"), "spoon": ("معلقة", "معالق"), "bowl": ("طبق", "أطباق"),
    "banana": ("موزة", "موز"), "apple": ("تفاحة", "تفاح"), "sandwich": ("ساندوتش", "ساندوتشات"),
    "orange": ("برتقانة", "برتقان"), "broccoli": ("بروكلي", "بروكلي"), "carrot": ("جزرة", "جزر"),
    "hot dog": ("هوت دوج", "هوت دوج"), "pizza": ("بيتزا", "بيتزا"), "donut": ("دونات", "دونات"),
    "cake": ("كيكة", "كيك"), "chair": ("كرسي", "كراسي"), "couch": ("كنبة", "كنب"),
    "potted plant": ("زرعة", "زرع"), "bed": ("سرير", "سراير"), "dining table": ("ترابيزة", "ترابيزات"),
    "toilet": ("تواليت", "تواليتات"), "tv": ("تلفزيون", "تلفزيونات"), "laptop": ("لابتوب", "لابتوبات"),
    "mouse": ("ماوس", "ماوسات"), "remote": ("ريموت", "ريموتات"), "keyboard": ("كيبورد", "كيبوردات"),
    "cell phone": ("موبايل", "موبايلات"), "microwave": ("ميكروويف", "ميكروويفات"),
    "oven": ("فرن", "أفران"), "toaster": ("محمصة", "محمصات"), "sink": ("حوض", "أحواض"),
    "refrigerator": ("تلاجة", "تلاجات"), "book": ("كتاب", "كتب"), "clock": ("ساعة", "ساعات"),
    "vase": ("فازة", "فازات"), "scissors": ("مقص", "مقصات"), "teddy bear": ("دبدوب", "دباديب"),
    "hair drier": ("سشوار", "سشوارات"), "toothbrush": ("فرشة سنان", "فرش سنان"),
}

AR_NAMES.update({
    "door": ("باب", "أبواب"), "stairs": ("سلم", "سلالم"), "escalator": ("سلم كهربائي", "سلالم كهربائية"),
    "elevator": ("أسانسير", "أسانسيرات"), "window": ("شباك", "شبابيك"), "wall": ("حيطة", "حيطان"),
    "pole": ("عمود", "عواميد"), "crosswalk": ("عبور مشاة", "عبور مشاة"), "trash can": ("سلة زبالة", "سلات زبالة"),
    "wet floor sign": ("علامة أرض مبلولة", "علامات أرض مبلولة"), "keys": ("مفاتيح", "مفاتيح"),
    "glasses": ("نضارة", "نضارات"), "sunglasses": ("نضارة شمس", "نضارات شمس"), "wallet": ("محفظة", "محافظ"),
    "pen": ("قلم", "أقلام"), "notebook": ("كشكول", "كشاكيل"), "paper": ("ورقة", "ورق"),
    "medicine": ("دوا", "أدوية"), "pill bottle": ("علبة دوا", "علب دوا"), "pill box": ("علبة برشام", "علب برشام"),
    "charger": ("شاحن", "شواحن"), "headphones": ("سماعة", "سماعات"), "earphones": ("سماعة أذن", "سماعات أذن"),
    "watch": ("ساعة إيد", "ساعات إيد"), "ring": ("خاتم", "خواتم"), "credit card": ("كارت", "كروت"),
    "banknote": ("ورقة فلوس", "فلوس"), "coin": ("عملة معدن", "عملات معدن"), "plate": ("طبق", "أطباق"),
    "mug": ("مج", "مجات"), "glass of water": ("كوباية مية", "كوبايات مية"), "kettle": ("كاتل", "كاتلات"),
    "light switch": ("مفتاح نور", "مفاتيح نور"), "power outlet": ("فيشة كهربا", "فيش كهربا"),
    "door handle": ("أوكرة", "أوكر"), "faucet": ("حنفية", "حنفيات"), "towel": ("فوطة", "فوط"),
    "pillow": ("مخدة", "مخدات"), "blanket": ("بطانية", "بطاطين"), "shoes": ("جزمة", "جزم"),
    "slippers": ("شبشب", "شباشب"), "jacket": ("جاكيت", "جواكت"), "hat": ("طاقية", "طواقي"), "bag": ("شنطة", "شنط"),
    "box": ("علبة", "علب"), "cabinet": ("دولاب", "دواليب"), "shelf": ("رف", "رفوف"), "mirror": ("مراية", "مرايات"),
    "comb": ("مشط", "أمشاط"), "hairbrush": ("فرشة شعر", "فرش شعر"), "toothpaste": ("معجون سنان", "معاجين"),
    "soap": ("صابونة", "صابون"), "shampoo bottle": ("إزازة شامبو", "إزايز شامبو"), "tissue box": ("علبة مناديل", "علب مناديل"),
    "fire extinguisher": ("طفاية حريق", "طفايات حريق"), "sign": ("لافتة", "لافتات"), "fan": ("مروحة", "مراوح"),
    "air conditioner": ("تكييف", "تكييفات"), "lamp": ("أباجورة", "أباجورات"), "candle": ("شمعة", "شمع"),
    "calculator": ("آلة حاسبة", "آلات حاسبة"), "tablet": ("تابلت", "تابلتات"), "cane": ("عصاية", "عصيان"),
})

EN_PLURAL_IRREGULAR = {"person": "people", "mouse": "mice", "knife": "knives",
                       "sheep": "sheep", "skis": "skis", "scissors": "scissors", "keys": "keys",
                       "glasses": "glasses", "sunglasses": "sunglasses", "stairs": "stairs",
                       "shoes": "shoes", "slippers": "slippers", "headphones": "headphones",
                       "earphones": "earphones", "paper": "papers", "medicine": "medicines"}

# ---------- Currency (custom YOLO model classes) ----------
CURRENCY = {
    "0.25 pound m": ("25 piastre coin", "ربع جنيه معدن"),
    "0.25 pound": ("25 piastre note", "ربع جنيه ورق"),
    "0.50 pound m": ("50 piastre coin", "نص جنيه معدن"),
    "0.50 pound": ("50 piastre note", "نص جنيه ورق"),
    "1 pound m": ("1 pound coin", "جنيه معدن"),
    "1 pound": ("1 pound note", "جنيه ورق"),
    "10 pound new": ("new 10 pound note", "عشرة جنيه البلاستيك الجديدة"),
    "10 pound": ("10 pound note", "عشرة جنيه"),
    "100 pounds": ("100 pound note", "مية جنيه"),
    "20 pounds": ("20 pound note", "عشرين جنيه"),
    "200 pounds": ("200 pound note", "ميتين جنيه"),
    "5 pounds": ("5 pound note", "خمسة جنيه"),
    "50 pounds": ("50 pound note", "خمسين جنيه"),
    "20 pound new": ("new 20 pound note", "عشرين جنيه البلاستيك الجديدة"),
}
CURRENCY_VALUE = {"0.25 pound m": 0.25, "0.25 pound": 0.25, "0.50 pound m": 0.5, "0.50 pound": 0.5,
                  "1 pound m": 1, "1 pound": 1, "10 pound new": 10, "10 pound": 10,
                  "100 pounds": 100, "20 pounds": 20, "200 pounds": 200, "5 pounds": 5, "50 pounds": 50,
                  "20 pound new": 20}

EN_NUM = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
AR_NUM = ["صفر", "واحد", "اتنين", "تلاتة", "أربعة", "خمسة", "ستة", "سبعة", "تمانية", "تسعة", "عشرة"]


def is_ar(lang):
    return str(lang or "").lower().startswith("ar")


def num(n, lang):
    table = AR_NUM if is_ar(lang) else EN_NUM
    return table[n] if 0 <= n < len(table) else str(n)


def object_name(label, count=1, lang="en"):
    if is_ar(lang):
        one, many = AR_NAMES.get(label, (label, label))
        return one if count == 1 else many
    if count == 1:
        return label
    if label in EN_PLURAL_IRREGULAR:
        return EN_PLURAL_IRREGULAR[label]
    last = label.split(" ")[-1]
    return label + ("es" if last.endswith(("s", "sh", "ch", "x")) else "s")


def counted(label, count, lang):
    """'a chair' / 'two chairs' / 'كرسي' / 'تلاتة كراسي'."""
    if is_ar(lang):
        if count == 1:
            return object_name(label, 1, lang)
        if count == 2:
            return f"اتنين {object_name(label, 2, lang)}"
        return f"{num(count, lang)} {object_name(label, count, lang)}"
    if count == 1:
        if label in ("keys", "glasses", "sunglasses", "stairs", "shoes", "slippers", "headphones", "earphones"):
            return f"some {label}" if label != "stairs" else "stairs"
        article = "an" if label[0] in "aeiou" else "a"
        return f"{article} {label}"
    return f"{num(count, lang)} {object_name(label, count, lang)}"


# ---------------------------------------
# Distance in steps
# ---------------------------------------
# A blind user counts steps, not metres. Each request carries the user's choice
# ("steps" or "meters") and their own step length; the default step is 65 cm
# (a typical adult walking with a cane; about 41% of body height).
from contextvars import ContextVar

DEFAULT_STEP_M = 0.65
ARM_REACH_M = 0.6           # closer than this: "within reach of your hand"
MAX_STEPS = 25              # further than this: just "far"
_units = ContextVar("units", default=("steps", DEFAULT_STEP_M))


def set_units(unit=None, step_cm=None):
    """Called once per request with what the page sent."""
    unit = "meters" if str(unit or "").lower().startswith("m") else "steps"
    try:
        step = float(step_cm) / 100
        step = step if 0.3 <= step <= 1.2 else DEFAULT_STEP_M
    except (TypeError, ValueError):
        step = DEFAULT_STEP_M
    _units.set((unit, step))


def steps_away(distance_m, lang, step_m=None):
    """'about 3 steps away' / 'على بعد حوالي 3 خطوات' (None if unknown)."""
    if distance_m is None:
        return None
    ar = is_ar(lang)
    if distance_m < ARM_REACH_M:
        return "قريب منك، تقدر توصله بإيدك" if ar else "within reach of your hand"
    n = distance_m / (step_m or _units.get()[1])
    if n > MAX_STEPS:
        return "بعيد" if ar else "far away"
    n = max(1, int(n + 0.5))
    if ar:
        word = "خطوة واحدة" if n == 1 else "خطوتين" if n == 2 else f"{n} خطوات" if n <= 10 else f"{n} خطوة"
        return f"على بعد حوالي {word}"
    return f"about {n} {'step' if n == 1 else 'steps'} away"


def where(info, lang):
    """'at 12 o'clock, about 3 steps away' / 'عند الساعة 12، على بعد حوالي 3 خطوات'.
    In metres when the user chose metres."""
    clock = info.get("clock")
    unit = _units.get()[0]
    if is_ar(lang):
        parts = [f"عند الساعة {clock}"] if clock else []
    else:
        parts = [f"at {clock} o'clock"] if clock else []
    if unit == "steps" and info.get("distance_m") is not None:
        parts.append(steps_away(info["distance_m"], lang))
        return ("، " if is_ar(lang) else ", ").join(parts)
    dist = info.get("distance")
    if is_ar(lang):
        if dist == "close":
            parts.append("قريب جدًا منك")
        elif dist == "far":
            parts.append("بعيد")
        elif dist:
            parts.append(f"على بعد حوالي {_fmt(dist)} متر")
        return "، ".join(parts)
    if dist == "close":
        parts.append("very close to you")
    elif dist == "far":
        parts.append("far away")
    elif dist:
        parts.append(f"about {_fmt(dist)} {'meter' if dist == 1 else 'meters'} away")
    return ", ".join(parts)


def _fmt(d):
    return str(int(d)) if float(d).is_integer() else str(d)


def join(items, lang):
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if is_ar(lang):
        return "، ".join(items[:-1]) + " و" + items[-1]
    return ", ".join(items[:-1]) + " and " + items[-1]


def sentence(text):
    text = text.strip()
    if not text:
        return text
    if text[0].isascii():
        text = text[0].upper() + text[1:]
    return text if text.endswith((".", "!", "?", "؟")) else text + "."


# ---------- Fixed messages ----------
MESSAGES = {
    "nothing_detected": ("I don't see anything I recognise.", "مش شايف حاجة أعرفها."),
    "no_currency": ("I don't see any money. Hold the note closer and flat.", "مش شايف فلوس. قرّب الورقة وخليها مفرودة."),
    "no_text": ("Nothing to read.", "مفيش كلام أقراه."),
    "no_summary": ("Nothing to summarise.", "مفيش كلام ألخصه."),
    "no_faces": ("No faces detected.", "مش شايف وشوش."),
    "not_installed": ("This mode is not installed on this device.", "الخاصية دي مش متسطبة على الجهاز ده."),
    "mode_error": ("Sorry, this mode is not working right now.", "معلش، الخاصية دي مش شغالة دلوقتي."),
    "choose_mode": ("Please choose a mode first.", "اختار خاصية الأول."),
    "no_image": ("No image received.", "موصلتش صورة."),
    "what_to_find": ("What should I look for?", "أدور على إيه؟"),
    "need_name": ("Please say the person's name, for example: remember Mona.", "قول اسم الشخص، مثلًا: احفظ منى."),
    "known_none": ("I don't know anyone yet.", "لسه مش عارف حد."),
}


def msg(key, lang):
    en, ar = MESSAGES[key]
    return ar if is_ar(lang) else en
