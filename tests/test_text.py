from mizan.text import normalize_arabic, normalized_label, split_sentences


def test_normalize_arabic_is_stable():
    assert normalize_arabic("إِنَّ الحقيقةَ… https://example.com") == "ان الحقيقه"


def test_labels_are_canonicalized():
    assert normalized_label("FALSE") == "False"
    assert normalized_label("Partly-False") == "Partly-false"


def test_sentence_split_keeps_reviewable_text():
    assert split_sentences("جملة أولى. جملة ثانية؟") == ["جملة أولى.", "جملة ثانية؟"]
