"""Read printed text with Tesseract OCR.

Improvements vs. the original version:
 * Small images are enlarged and binarised (Otsu) - much better on phone photos.
 * Words Tesseract is unsure about (confidence < 55) are dropped, so a photo
   with no text says "Nothing to read" instead of reading out random symbols.
 * Arabic is used automatically when the Arabic language pack is installed.
 * Tesseract is found automatically (PATH or the default Windows folder).
"""
import os
import shutil

import cv2
import pytesseract

WINDOWS_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if shutil.which("tesseract") is None and os.path.exists(WINDOWS_PATH):
    pytesseract.pytesseract.tesseract_cmd = WINDOWS_PATH

NOTHING = "Nothing To Read"
# Language data shipped with the project (English + Arabic), so Arabic works
# without installing anything into Program Files.
TESSDATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tessdata")
# Point Tesseract at the project's own language files with TESSDATA_PREFIX (an environment setting).
# A "--tessdata-dir" flag broke on Windows: the quotes around the path were passed on literally.
if os.path.exists(os.path.join(TESSDATA, "eng.traineddata")):
    os.environ["TESSDATA_PREFIX"] = TESSDATA
TD_CONFIG = ""
MIN_CONF = 55

_langs = None


def available_languages():
    global _langs
    if _langs is None:
        try:
            _langs = set(pytesseract.get_languages(config=TD_CONFIG))
        except Exception:
            _langs = {"eng"}
    return _langs


def preprocess_adaptive(image):
    """Second attempt for uneven light / screens with glare."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if max(h, w) < 1600:
        scale = 1600 / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 15)


def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if max(h, w) < 1600:                                  # enlarge small text
        scale = 1600 / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if binary.mean() < 127:                               # light text on dark background
        binary = 255 - binary
    return binary


def ocr_image_to_text(image_path, lang=None):
    image = cv2.imread(image_path) if isinstance(image_path, str) else image_path
    if image is None:
        return NOTHING
    # Signs and papers in Egypt mix Arabic and English: read both when possible
    langs = "eng"
    if "ara" in available_languages():
        langs = "ara+eng" if lang and str(lang).startswith("ar") else "eng+ara"
    best_text, best_words = "", []
    for prep, psm, needed in ((preprocess, 3, 2), (preprocess_adaptive, 6, 3)):
        text, words = _read(prep(image), langs, psm)
        if len(words) >= needed and len(words) > len(best_words):
            best_text, best_words = text, words
        if len(best_words) >= 6:          # clearly readable: no need for a second pass
            break
    return best_text if best_words else NOTHING


def _read(binary, langs, psm):
    data = pytesseract.image_to_data(binary, lang=langs, config=f"{TD_CONFIG} --oem 3 --psm {psm}",
                                     output_type=pytesseract.Output.DICT)
    lines = {}
    for i, word in enumerate(data["text"]):
        word = word.strip()
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1
        if word and conf >= MIN_CONF and any(c.isalnum() for c in word):
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append(word)
    # keep lines that contain at least one real word (3+ letters): drops glare/noise
    kept = [" ".join(ws) for _, ws in sorted(lines.items())
            if any(sum(c.isalpha() for c in w) >= 3 for w in ws)]
    text = "\n".join(kept)
    return text, [w for w in text.split() if sum(c.isalpha() for c in w) >= 3]


def test_ocr():
    print(ocr_image_to_text("static/logo.png"))
