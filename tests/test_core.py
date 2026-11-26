"""Unit tests: geometry, language, finder words, OCR, face decisions, guide mode."""
import numpy as np
import cv2
import pytest
from PIL import Image, ImageDraw, ImageFont

from core import i18n, vision


# ---------- geometry ----------
@pytest.mark.parametrize("x1,x2,expected", [(0, 100, 11), (590, 690, 12), (1180, 1280, 1)])
def test_clock_direction(x1, x2, expected):
    assert vision.clock_direction(x1, x2, 1280) == expected


def test_distance_shrinks_as_box_grows():
    far = vision.estimate_distance(0.45, 100, 1280)
    near = vision.estimate_distance(0.45, 400, 1280)
    assert far > near > 0


def test_rounded_distance():
    assert vision.rounded_distance(0.5) == "close"
    assert vision.rounded_distance(1.3) == 1.5
    assert vision.rounded_distance(7.4) == 7.0
    assert vision.rounded_distance(40) == "far"


def test_every_yolo_class_has_a_real_size():
    from core.models import object_model
    missing = [n for n in object_model().names.values() if n not in vision.REAL_WIDTH_M]
    assert not missing, missing


# ---------- language ----------
def test_english_and_arabic_counts():
    assert i18n.counted("chair", 1, "en") == "a chair"
    assert i18n.counted("person", 3, "en") == "three people"
    assert i18n.counted("chair", 3, "ar") == "تلاتة كراسي"


def test_every_yolo_class_has_an_arabic_name():
    from core.models import object_model
    missing = [n for n in object_model().names.values() if n not in i18n.AR_NAMES]
    assert not missing, missing


def test_where_sentence():
    info = {"clock": 1, "distance": 2.0}
    assert i18n.where(info, "en") == "at 1 o'clock, about 2 meters away"
    assert "الساعة 1" in i18n.where(info, "ar")


# ---------- finder ----------
@pytest.mark.parametrize("said,expected", [
    ("Bottle", "bottle"), ("bottles", "bottle"), ("the bottle", "bottle"), ("a phone", "cell phone"),
    ("sofa", "couch"), ("tv", "tv"), ("plant", "potted plant"), ("إزازة", "bottle"),
    ("الموبايل", "cell phone"), ("ابحث عن كرسي", "chair"), ("keys", "keys"), ("dragon", None), ("", None)])
def test_finder_understands_words(said, expected):
    from finder_and_depth_estimation import normalise
    assert normalise(said) == expected


def test_finder_never_crashes_on_any_class():
    from core.models import object_model
    from finder_and_depth_estimation import find_object
    img = cv2.imread("images/car_person.jpg")
    for name in object_model().names.values():
        text, _ = find_object(img, name)
        assert isinstance(text, str) and text


# ---------- OCR ----------
def _text_image(lines):
    img = Image.new("RGB", (900, 120 * len(lines)), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 44)
    except OSError:
        font = ImageFont.load_default()
    for i, line in enumerate(lines):
        d.text((30, 30 + 110 * i), line, fill="black", font=font)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def test_ocr_reads_printed_text():
    from ocr_function import ocr_image_to_text
    text = ocr_image_to_text(_text_image(["Take one tablet", "twice a day"]))
    assert "tablet" in text.lower() and "twice" in text.lower()


def test_ocr_says_nothing_on_photos_without_text():
    from ocr_function import NOTHING, ocr_image_to_text
    assert ocr_image_to_text(cv2.imread("images/cat_dog.jpg")) == NOTHING


# ---------- face decisions ----------
def test_decide_rules():
    import face_function as F
    assert F.decide([("Ahmed", 0.60), ("Saif", 0.20)])[2] == "sure"
    assert F.decide([("Ahmed", 0.417), ("Saif", 0.401)])[2] == "unknown"   # too close to call
    assert F.decide([("Ahmed", 0.39), ("Saif", 0.27)])[2] == "maybe"
    assert F.decide([("Ahmed", 0.30)])[2] == "unknown"
    assert F.decide([])[2] == "unknown"


def test_enroll_rejects_photo_without_face(empty_faces):
    text, info = empty_faces.enroll([cv2.imread("images/car.jpg")], "Mona")
    assert info["saved"] == 0 and "face" in text.lower()


def test_enroll_needs_a_name(empty_faces):
    text, info = empty_faces.enroll([cv2.imread("images/car.jpg")], "  ")
    assert info["saved"] == 0


def test_forget_unknown_person(empty_faces):
    assert "don't know" in empty_faces.forget("Nobody")


def test_pet_face_is_not_a_person(empty_faces):
    text, people = empty_faces.recognize(cv2.imread("images/cat_dog.jpg"))
    assert people == []


# ---------- describe caption cleaning ----------
def test_caption_cleaning():
    pytest.importorskip("transformers")
    from describe_function_2 import clean_caption
    assert clean_caption("a picture of arafed man sitting") == "Man sitting."
    assert clean_caption("a magnifying magnifying icon") == "A magnifying icon."


# ---------- guide mode ----------
def test_guide_reports_newcomer_then_approach(empty_faces):
    from guide import GuideSession
    person = cv2.imread("images/car_person.jpg")
    bg = np.full((720, 1280, 3), 200, np.uint8)

    def frame(scale):
        im = bg.copy(); h, w = person.shape[:2]; ph = int(720 * scale); pw = int(w * ph / h)
        p = cv2.resize(person, (pw, ph)); x = 640 - pw // 2
        im[720 - ph:720, max(0, x):max(0, x) + pw] = p[:, :min(pw, 1280 - max(0, x))]
        return im

    g, t, said = GuideSession(), 0.0, []
    for sc in (0.3, 0.3, 0.4, 0.55, 0.75, 0.95):
        events, _ = g.step(frame(sc), "en", now=t)
        said += [e["type"] for e in events]
        t += 1.2
    assert "newcomer" in said and "approach" in said


# ---------- open vocabulary & Arabic OCR (version 2.1) ----------
def test_open_vocabulary_words():
    from finder_and_depth_estimation import normalise
    assert normalise("my keys") == "keys"
    assert normalise("the door") == "door"
    assert normalise("السلم") == "stairs"
    assert normalise("socket") == "power outlet"


def test_find_door_with_open_vocabulary():
    from finder_and_depth_estimation import find_object
    text, dets = find_object(cv2.imread("images/window_door.jpg"), "door")
    assert "door" in text.lower() and "o'clock" in text


def test_ocr_reads_arabic():
    from ocr_function import available_languages, ocr_image_to_text
    if "ara" not in available_languages():
        pytest.skip("Arabic language data not installed")
    img = Image.new("RGB", (1100, 160), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 56)
        d.text((1060, 40), "خذ قرص واحد مرتين يوميا", fill="black", font=font, direction="rtl", anchor="ra")
    except Exception:
        pytest.skip("no Arabic text rendering on this machine")
    text = ocr_image_to_text(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), "ar")
    assert "مرتين" in text


def test_distance_in_steps():
    from core import i18n
    i18n.set_units("steps", 65)
    assert i18n.steps_away(0.4, "en") == "within reach of your hand"
    assert i18n.steps_away(1.3, "en") == "about 2 steps away"
    assert i18n.steps_away(1.3, "ar") == "على بعد حوالي خطوتين"
    assert i18n.steps_away(0.7, "ar") == "على بعد حوالي خطوة واحدة"
    assert i18n.steps_away(20, "en") == "far away"
    assert i18n.where({"clock": 12, "distance_m": 2.0, "distance": 2.0}, "en") == "at 12 o'clock, about 3 steps away"
    i18n.set_units("steps", 80)           # a taller person takes longer steps
    assert i18n.where({"clock": 12, "distance_m": 2.0, "distance": 2.0}, "en") == "at 12 o'clock, about 3 steps away"
    i18n.set_units("steps", "999")        # nonsense falls back to the default
    assert i18n.steps_away(1.3, "en") == "about 2 steps away"
    i18n.set_units("meters", None)
    assert i18n.where({"clock": 1, "distance_m": 2.0, "distance": 2.0}, "en") == "at 1 o'clock, about 2 meters away"
    i18n.set_units("steps", 65)


def test_steps_sent_by_the_page(client):
    import cv2
    with open("images/car_person.jpg", "rb") as f:
        r = client.post("/detect", data={"select_mode": "object", "lang": "en", "unit": "steps", "step_cm": "65",
                                         "image": (f, "x.jpg")}, content_type="multipart/form-data")
    text = r.get_json()[0]["result"]
    assert "step" in text or "reach" in text, text
