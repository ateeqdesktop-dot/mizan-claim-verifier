"""Arabic text normalization utilities used consistently at train and inference time."""

from __future__ import annotations

import re
import unicodedata

# Arabic diacritics, Quranic marks, and tatweel.
_ARABIC_MARKS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_NON_TEXT = re.compile(r"[^\w\s\u0600-\u06FF]", re.UNICODE)


def normalize_arabic(text: object) -> str:
    """Return a conservative normalized form suitable for retrieval/classification.

    The function deliberately does not stem or aggressively remove stop words. Such
    operations can remove names, negation, or dates that are informative in fact
    checking. The original input must be stored separately by callers.
    """

    if text is None:
        return ""
    value = unicodedata.normalize("NFKC", str(text))
    value = _URL.sub(" ", value)
    value = _ARABIC_MARKS.sub("", value)
    value = value.replace("ـ", " ")
    value = value.replace("إ", "ا").replace("أ", "ا").replace("آ", "ا")
    value = value.replace("ى", "ي").replace("ة", "ه")
    value = value.replace("ؤ", "و").replace("ئ", "ي")
    value = value.replace("گ", "ك").replace("ڤ", "ف").replace("پ", "ب").replace("چ", "ج")
    value = _NON_TEXT.sub(" ", value)
    value = value.lower()
    return _WHITESPACE.sub(" ", value).strip()


def split_sentences(text: object) -> list[str]:
    """Split Arabic/Latin article text into reviewable sentence candidates."""

    if text is None:
        return []
    raw = re.split(r"(?<=[.!?؟؛])\s+|\n+", str(text))
    return [sentence.strip() for sentence in raw if sentence.strip()]


def normalized_label(value: object) -> str:
    """Canonicalize the label variants observed in AraFacts exports."""

    label = str(value).strip().lower().replace("_", "-")
    mapping = {
        "false": "False",
        "partly-false": "Partly-false",
        "partly false": "Partly-false",
        "true": "True",
        "sarcasm": "Sarcasm",
        "unverifiable": "Unverifiable",
    }
    return mapping.get(label, str(value).strip())
