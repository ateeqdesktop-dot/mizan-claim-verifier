from pathlib import Path

import pandas as pd

from mizan.data import chronological_split, stratified_split


def make_frame() -> pd.DataFrame:
    rows = []
    for index, label in enumerate(["False", "Partly-false", "True", "Sarcasm", "False", "Partly-false", "True", "Sarcasm"]):
        rows.append(
            {
                "ClaimID": f"C{index}",
                "claim": f"ادعاء {index}",
                "description": "",
                "source": "test",
                "date": f"2020-01-{index + 1:02d}",
                "normalized_label": label,
                "normalized_category": "Social",
                "content": f"مقال تحقق للادعاء {index}.",
                "claim_normalized": f"ادعاء {index}",
                "content_normalized": f"مقال تحقق للادعاء {index}",
                "date_parsed": pd.Timestamp(f"2020-01-{index + 1:02d}", tz="UTC"),
            }
        )
    return pd.DataFrame(rows)


def test_chronological_split_is_non_empty_and_ordered():
    splits = chronological_split(make_frame(), train_fraction=0.5, validation_fraction=0.25)
    assert len(splits.train) > 0 and len(splits.validation) > 0 and len(splits.test) > 0
    assert splits.train["date_parsed"].max() <= splits.test["date_parsed"].min()


def test_stratified_split_preserves_classes():
    train, test = stratified_split(make_frame(), test_size=0.50, random_state=7)
    assert set(test["normalized_label"]) == {"False", "Partly-false", "True", "Sarcasm"}
