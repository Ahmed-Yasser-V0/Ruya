# ---------------------------------------
# Smart Glasses - Flask server
# ---------------------------------------
# Routes
#   GET  /               the web interface
#   POST /detect         one-shot modes (object, find, currency, describe, text,
#                        summarize, face, add_face, forget_face, list_faces)
#   POST /face/enroll    guided enrollment: several photos of one person at once
#   POST /guide          Guide mode: one frame of the continuous stream
#   POST /guide/reset    forget Guide mode's memory (when it is switched off)
#   GET  /health         which modes are available on this machine
#
# Every answer is JSON: [{"result": text, "mode": ..., "detections": [...], "lang": ...}]
# "detections" holds normalised boxes (0..1) that the page draws over the camera.
# ---------------------------------------
import os
import shutil
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from core import i18n
from core.models import object_model, path

import detection_function as detection
import face_function as faces
import finder_and_depth_estimation as finder
import guide

app = Flask(__name__, template_folder=path("templates"), static_folder=path("static"))
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024     # reject huge uploads
MAX_IMAGE_BYTES = 16 * 1024 * 1024
app.config["TEMPLATES_AUTO_RELOAD"] = True               # page edits apply on browser refresh
UPLOAD_FOLDER = path("uploads")
KEEP_UPLOADS = True     # keep today's frames for debugging (older days are deleted)

# ---------------------------------------
# Optional modes: if a library/model is missing on this device the app still
# starts and that mode answers "not installed" instead of crashing everything.
# ---------------------------------------
AVAILABLE = {}
try:
    from describe_function_2 import generate_caption_from_image, test_describe
    AVAILABLE["describe"] = True
except Exception as e:
    print(f"⚠️  Description mode disabled: {e}")
    generate_caption_from_image, test_describe = None, None
try:
    from ocr_function import NOTHING, ocr_image_to_text, test_ocr
    AVAILABLE["text"] = True
except Exception as e:
    print(f"⚠️  Text (OCR) mode disabled: {e}")
    ocr_image_to_text, test_ocr, NOTHING = None, None, "Nothing To Read"
try:
    from summarize_function import summarize_text, test_summarize
    AVAILABLE["summarize"] = True
except Exception as e:
    print(f"⚠️  Summarize mode disabled: {e}")
    summarize_text, test_summarize = None, None


# ---------------------------------------
# Start-up checks (also loads every model once, so the first request is fast)
# ---------------------------------------
def test_program_functions():
    tests = [
        ("Detection Mode", detection.test_detection),
        ("Finder Mode", finder.test_find),
        ("Face Recognition Mode", faces.test_face),
        ("Description Mode", test_describe),
        ("OCR Text Mode", test_ocr),
        ("Summarize Text Mode", test_summarize),
    ]
    for name, test in tests:
        print(f"Testing the {name} ............")
        if test is None:
            print("Skipped (not installed) ⚠️")
            continue
        try:
            test()
            print("Done ✔️")
        except Exception as e:
            key = {"Description Mode": "describe", "OCR Text Mode": "text", "Summarize Text Mode": "summarize"}.get(name)
            if key:
                AVAILABLE[key] = False
            print(f"Not working on this device ⚠️  ({e})")
    print("---------------------------------------------------")
    print("The Program Is Ready ✔️ ---------------------------")
    print("Open http://localhost:5001 in Chrome")
    print("---------------------------------------------------")


# ---------------------------------------
# Helpers
# ---------------------------------------
def read_image(file_storage):
    data = np.frombuffer(file_storage.read(MAX_IMAGE_BYTES + 1), dtype=np.uint8)
    if data.size > MAX_IMAGE_BYTES:      # photos stay limited; only CCTV accepts big files
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def save_frame(image):
    """Keep today's frames in uploads/<date>/ (older days are removed)."""
    if not KEEP_UPLOADS:
        return
    today = datetime.now().strftime("%Y-%m-%d")
    folder = os.path.join(UPLOAD_FOLDER, today)
    os.makedirs(folder, exist_ok=True)
    for old in os.listdir(UPLOAD_FOLDER):
        old_path = os.path.join(UPLOAD_FOLDER, old)
        if os.path.isdir(old_path) and old != today:
            shutil.rmtree(old_path, ignore_errors=True)
    cv2.imwrite(os.path.join(folder, datetime.now().strftime("%Y-%m-%d_%H-%M-%S_frame.jpg")), image)


def answer(text, mode, lang, detections=None, **extra):
    payload = {"result": text, "mode": mode, "lang": lang, "detections": detections or []}
    payload.update(extra)
    return jsonify([payload])


def not_installed(mode, lang):
    return answer(i18n.msg("not_installed", lang), mode, lang)


# ---------------------------------------
# Routes
# ---------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", names=object_model().names)


@app.route("/health")
def health():
    return jsonify({"ok": True, "modes": {"object": True, "find": True, "currency": True, "face": True,
                                          "describe": AVAILABLE.get("describe", False),
                                          "text": AVAILABLE.get("text", False),
                                          "summarize": AVAILABLE.get("summarize", False)}})


@app.route("/detect", methods=["POST"])
def detect():
    i18n.set_units(request.form.get("unit"), request.form.get("step_cm"))
    mode = request.form.get("select_mode") or ""
    lang = "ar" if i18n.is_ar(request.form.get("lang")) else "en"
    if "image" not in request.files:
        return answer(i18n.msg("no_image", lang), mode or "object", lang)
    image = read_image(request.files["image"])
    if image is None:
        return answer(i18n.msg("no_image", lang), mode, lang)
    save_frame(image)

    try:
        if mode in ("object", "currency"):
            text, dets = detection.detect(image, mode, lang)
            return answer(text, mode, lang, dets)
        if mode == "find":
            text, dets = finder.find_object(image, request.form.get("object_to_be_found", ""), lang)
            return answer(text, mode, lang, dets)
        if mode == "face":
            text, dets = faces.recognize(image, lang)
            return answer(text, mode, lang, dets)
        if mode == "add_face":
            text, info = faces.enroll([image], request.form.get("person_name", ""), lang)
            return answer(text, mode, lang, **info)
        if mode == "forget_face":
            return answer(faces.forget(request.form.get("person_name", ""), lang), mode, lang)
        if mode == "list_faces":
            return answer(faces.known_people_sentence(lang), mode, lang)
        if mode == "describe":
            if not AVAILABLE.get("describe"):
                return not_installed(mode, lang)
            caption = generate_caption_from_image(image) or ("مش قادر أوصف الصورة." if lang == "ar" else "I can't describe this scene.")
            # BLIP writes English; the page translates it when Arabic is selected
            return answer(caption, mode, lang, translate=(lang == "ar"))
        if mode == "text":
            if not AVAILABLE.get("text"):
                return not_installed(mode, lang)
            text = ocr_image_to_text(image, lang)
            return answer(i18n.msg("no_text", lang) if text == NOTHING else " ".join(text.split()), mode, lang)
        if mode == "summarize":
            if not AVAILABLE.get("summarize"):
                return not_installed(mode, lang)
            text = summarize_text(image, lang)
            if text == "Nothing To Summarize":
                return answer(i18n.msg("no_summary", lang), mode, lang)
            short = text.startswith("الكلام قصير")
            return answer(text, mode, lang, translate=(lang == "ar" and not short))
        return answer(i18n.msg("choose_mode", lang), mode, lang)
    except Exception as e:
        print(f"⚠️  Error in mode '{mode}': {e}")
        return answer(i18n.msg("mode_error", lang), mode, lang)


@app.route("/face/enroll", methods=["POST"])
def face_enroll():
    i18n.set_units(request.form.get("unit"), request.form.get("step_cm"))
    lang = "ar" if i18n.is_ar(request.form.get("lang")) else "en"
    images = [img for img in (read_image(f) for f in request.files.getlist("images")) if img is not None]
    if not images:
        return answer(i18n.msg("no_image", lang), "add_face", lang)
    try:
        text, info = faces.enroll(images, request.form.get("person_name", ""), lang)
        return answer(text, "add_face", lang, **info)
    except Exception as e:
        print(f"⚠️  Error in enrollment: {e}")
        return answer(i18n.msg("mode_error", lang), "add_face", lang)


@app.route("/guide", methods=["POST"])
def guide_step():
    i18n.set_units(request.form.get("unit"), request.form.get("step_cm"))
    lang = "ar" if i18n.is_ar(request.form.get("lang")) else "en"
    if "image" not in request.files:
        return jsonify({"events": [], "detections": []})
    image = read_image(request.files["image"])
    if image is None:
        return jsonify({"events": [], "detections": []})
    try:
        events, dets = guide.get_session(request.form.get("session", "default")).step(image, lang)
        return jsonify({"events": events, "detections": dets})
    except Exception as e:
        print(f"⚠️  Error in guide mode: {e}")
        return jsonify({"events": [], "detections": [], "error": True})


@app.route("/guide/reset", methods=["POST"])
def guide_reset():
    guide.reset_session(request.form.get("session", "default"))
    return jsonify({"ok": True})


# ---------------------------------------
# Run The App
# ---------------------------------------
if __name__ == "__main__":
    test_program_functions()
    host, port = "0.0.0.0", 5001

    # Windows + Python 3.14: looking up this computer's network name can crash
    # ("idna codec ... BIDI requirement"). The name isn't needed, so fall back safely.
    import socket
    _original_getfqdn = socket.getfqdn

    def _safe_getfqdn(name=""):
        try:
            return _original_getfqdn(name)
        except Exception:
            return name or "localhost"
    socket.getfqdn = _safe_getfqdn

    import threading
    import webbrowser
    threading.Timer(2, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    # debug=False: the Werkzeug debugger must never be reachable from the network
    app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)
