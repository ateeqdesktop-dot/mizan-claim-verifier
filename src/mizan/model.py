"""Trainable text classifier and artifact metadata for Mizan."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .text import normalize_arabic


def make_pair_text(claim: object, content: object) -> str:
    """Create a stable claim/evidence text pair without exposing raw URLs."""

    return f"{normalize_arabic(claim)} [sep] {normalize_arabic(content)}".strip()


@dataclass
class MizanClassifier:
    pipeline: Pipeline | None = None
    labels: list[str] = field(default_factory=list)
    version: str = "0.1.0"
    training_rows: int = 0
    include_evidence: bool = False

    def _build_pipeline(self) -> Pipeline:
        features = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=1,
                        max_df=0.98,
                        sublinear_tf=True,
                        max_features=65000,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=1,
                        sublinear_tf=True,
                        max_features=90000,
                    ),
                ),
            ]
        )
        return Pipeline(
            [
                ("features", features),
                (
                    "classifier",
                    LogisticRegression(
                        C=2.0,
                        class_weight="balanced",
                        max_iter=700,
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        )

    def fit(self, frame: pd.DataFrame) -> "MizanClassifier":
        required = {"claim", "content", "normalized_label"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"training frame missing columns: {sorted(missing)}")
        x = [
            make_pair_text(claim, content) if self.include_evidence else normalize_arabic(claim)
            for claim, content in zip(frame["claim"], frame["content"])
        ]
        y = frame["normalized_label"].astype(str).tolist()
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(x, y)
        self.labels = [str(label) for label in self.pipeline.named_steps["classifier"].classes_]
        self.training_rows = len(frame)
        return self

    def _ensure_fitted(self) -> None:
        if self.pipeline is None:
            raise RuntimeError("classifier has not been fitted")

    def predict(self, frame: pd.DataFrame) -> list[str]:
        self._ensure_fitted()
        x = [
            make_pair_text(claim, content) if self.include_evidence else normalize_arabic(claim)
            for claim, content in zip(frame["claim"], frame["content"])
        ]
        return [str(value) for value in self.pipeline.predict(x)]

    def predict_proba(self, frame: pd.DataFrame) -> list[dict[str, float]]:
        self._ensure_fitted()
        x = [
            make_pair_text(claim, content) if self.include_evidence else normalize_arabic(claim)
            for claim, content in zip(frame["claim"], frame["content"])
        ]
        probabilities = self.pipeline.predict_proba(x)
        classes = [str(value) for value in self.pipeline.named_steps["classifier"].classes_]
        return [{label: float(probability) for label, probability in zip(classes, row)} for row in probabilities]

    def metadata(self) -> dict[str, Any]:
        return {
            "model_version": self.version,
            "training_rows": self.training_rows,
            "labels": self.labels,
            "model_type": "word+character TF-IDF with balanced Logistic Regression",
            "include_evidence": self.include_evidence,
        }

    def save(self, path: str | Path) -> None:
        self._ensure_fitted()
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "MizanClassifier":
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"artifact at {path} is not a {cls.__name__}")
        return model
