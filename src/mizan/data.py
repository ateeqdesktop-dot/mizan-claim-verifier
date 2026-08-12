"""Dataset loading, cleaning, and leakage-aware splitting for AraFacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .text import normalize_arabic, normalized_label

REQUIRED_CLAIM_COLUMNS = {
    "ClaimID",
    "claim",
    "description",
    "source",
    "date",
    "normalized_label",
    "normalized_category",
}
REQUIRED_CONTENT_COLUMNS = {"ClaimID", "content"}
LABELS = ["False", "Partly-false", "True", "Sarcasm", "Unverifiable"]


@dataclass(frozen=True)
class DatasetSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def _check_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def load_arafacts(claims_path: str | Path, content_path: str | Path) -> pd.DataFrame:
    """Load and join AraFacts claims with verification article content."""

    claims = pd.read_csv(claims_path)
    content = pd.read_csv(content_path)
    _check_columns(claims, REQUIRED_CLAIM_COLUMNS, "claims CSV")
    _check_columns(content, REQUIRED_CONTENT_COLUMNS, "content CSV")

    claims = claims.copy()
    content = content[["ClaimID", "content"]].copy()
    claims["normalized_label"] = claims["normalized_label"].map(normalized_label)
    claims["claim"] = claims["claim"].fillna("").astype(str)
    claims["description"] = claims["description"].fillna("").astype(str)
    claims["source"] = claims["source"].fillna("unknown").astype(str)
    claims["normalized_category"] = claims["normalized_category"].fillna("UNCATEGORIZED").astype(str)
    content["content"] = content["content"].fillna("").astype(str)

    frame = claims.merge(content, on="ClaimID", how="left", validate="one_to_one")
    frame["content"] = frame["content"].fillna("")
    frame["claim_normalized"] = frame["claim"].map(normalize_arabic)
    frame["content_normalized"] = frame["content"].map(normalize_arabic)
    frame["date_parsed"] = pd.to_datetime(frame["date"], errors="coerce", utc=True, format="mixed")
    frame = frame.drop_duplicates(subset=["claim_normalized", "content_normalized"], keep="first")
    frame = frame[frame["claim_normalized"].str.len() > 0].reset_index(drop=True)
    return frame


def chronological_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> DatasetSplits:
    """Split chronologically to make evaluation closer to future claims."""

    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be below one")

    ordered = frame.sort_values(["date_parsed", "ClaimID"], na_position="last").reset_index(drop=True)
    train_end = max(1, int(len(ordered) * train_fraction))
    validation_end = max(train_end + 1, int(len(ordered) * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, len(ordered) - 1)
    train = ordered.iloc[:train_end].reset_index(drop=True)
    validation = ordered.iloc[train_end:validation_end].reset_index(drop=True)
    test = ordered.iloc[validation_end:].reset_index(drop=True)
    if train.empty or validation.empty or test.empty:
        raise ValueError("chronological split produced an empty partition")
    return DatasetSplits(train=train, validation=validation, test=test)


def stratified_split(frame: pd.DataFrame, test_size: float = 0.20, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience split for unit tests and ablations."""

    train, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        stratify=frame["normalized_label"],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
