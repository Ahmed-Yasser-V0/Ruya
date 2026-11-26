"""Scene description with BLIP (Salesforce/blip-image-captioning-large).

Fixes vs. the original version:
 * OpenCV loads images as BGR, BLIP expects RGB: colours were swapped
   (a blue shirt looked orange to the model). The image is now converted.
 * `max_new_tokens` was given to the processor (ignored), so captions were cut
   at ~20 tokens and often repeated words ("a magnifying magnifying icon").
   Generation now uses beam search with a repetition guard.
 * The model is loaded once and shared.
"""
import re

import cv2
import transformers  # noqa: F401  (fail early if the library is missing)
from PIL import Image

from core.models import INFERENCE_LOCK, _load as _cached

MODEL_NAME = "Salesforce/blip-image-captioning-large"
PROMPT = "a picture of"
JUNK_WORDS = re.compile(r"\b(arafed|araffe|arafe|arrafed|there is|there are)\b\s*", re.I)


def _model():
    def load():
        from transformers import BlipForConditionalGeneration, BlipProcessor
        return (BlipProcessor.from_pretrained(MODEL_NAME),
                BlipForConditionalGeneration.from_pretrained(MODEL_NAME).eval())
    return _cached("blip", load)


def clean_caption(text):
    text = text.strip()
    if text.lower().startswith(PROMPT):
        text = text[len(PROMPT):].strip()
    text = JUNK_WORDS.sub("", text)
    words, out = text.split(), []
    for w in words:                       # drop immediate repeats ("a a", "magnifying magnifying")
        if not out or out[-1].lower() != w.lower():
            out.append(w)
    text = " ".join(out).strip(" ,.")
    return (text[:1].upper() + text[1:] + ".") if text else ""


def generate_caption_from_image(image_path):
    image = cv2.imread(image_path) if isinstance(image_path, str) else image_path
    if image is None:
        return ""
    rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    processor, model = _model()
    import torch
    with INFERENCE_LOCK, torch.inference_mode():
        inputs = processor(images=rgb, text=PROMPT, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=30, num_beams=3,
                             repetition_penalty=1.3, no_repeat_ngram_size=3)
    return clean_caption(processor.decode(out[0], skip_special_tokens=True))


def test_describe():
    _model()
    print(generate_caption_from_image("static/logo.png"))
