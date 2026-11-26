"""Summarise text read from a photo (txtai, distilbart-cnn-12-6).

The summariser is now loaded once (it was reloaded - 1.2 GB - on every request),
and short texts are read as-is instead of being "summarised" into nonsense.
"""
import txtai  # noqa: F401  (fail early if the library is missing)

from core.models import INFERENCE_LOCK, _load as _cached
from ocr_function import NOTHING, ocr_image_to_text

SHORT_TEXT_WORDS = 40


def _summary():
    from txtai.pipeline import Summary
    return _cached("summary", Summary)


def summarize_text(image_path, lang=None):
    text = ocr_image_to_text(image_path, lang)
    if text == NOTHING:
        return "Nothing To Summarize"
    words = text.split()
    if len(words) < SHORT_TEXT_WORDS:
        prefix = "الكلام قصير، وده هو: " if str(lang or "").startswith("ar") else "The text is short, so here it is: "
        return prefix + " ".join(words)
    with INFERENCE_LOCK:
        result = _summary()(" ".join(words))
    return result or "Nothing To Summarize"


def test_summarize():
    _summary()
