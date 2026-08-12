"""Lightweight, reproducible evidence retrieval over verification articles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .text import normalize_arabic, split_sentences


@dataclass
class EvidenceRetriever:
    """TF-IDF retriever that keeps article metadata for human review."""

    vectorizer: TfidfVectorizer | None = None
    matrix: Any = None
    documents: pd.DataFrame | None = None

    def fit(self, documents: pd.DataFrame) -> "EvidenceRetriever":
        required = {"ClaimID", "claim", "content", "content_normalized", "source", "date", "normalized_label"}
        missing = required.difference(documents.columns)
        if missing:
            raise ValueError(f"documents missing columns: {sorted(missing)}")
        self.documents = documents.reset_index(drop=True).copy()
        self.documents["retrieval_text"] = (
            self.documents["claim"].fillna("").map(normalize_arabic)
            + " "
            + self.documents["content_normalized"].fillna("")
        ).str.strip()
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.98,
            sublinear_tf=True,
            max_features=60000,
        )
        self.matrix = self.vectorizer.fit_transform(self.documents["retrieval_text"])
        return self

    def _ensure_fitted(self) -> None:
        if self.vectorizer is None or self.matrix is None or self.documents is None:
            raise RuntimeError("retriever has not been fitted")

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Retrieve top articles with cosine scores and compact metadata."""

        self._ensure_fitted()
        if top_k < 1:
            raise ValueError("top_k must be positive")
        normalized_query = normalize_arabic(query)
        if not normalized_query:
            return []
        query_vector = self.vectorizer.transform([normalized_query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        order = np.argsort(-scores)[:top_k]
        results: list[dict[str, Any]] = []
        for index in order:
            row = self.documents.iloc[int(index)]
            results.append(
                {
                    "claim_id": str(row["ClaimID"]),
                    "source": str(row["source"]),
                    "date": str(row["date"]),
                    "label": str(row["normalized_label"]),
                    "score": float(scores[index]),
                    "content": str(row["content"]),
                    "source_url": str(row.get("source_url", "")),
                    "evidence_urls": str(row.get("evidence_urls", "")),
                }
            )
        return results

    @staticmethod
    def rank_sentences(query: str, content: str, top_k: int = 2) -> list[dict[str, Any]]:
        """Return the most query-similar sentences for a human-readable rationale."""

        sentences = split_sentences(content)
        if not sentences or not normalize_arabic(query):
            return []
        corpus = [normalize_arabic(query)] + [normalize_arabic(sentence) for sentence in sentences]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        matrix = vectorizer.fit_transform(corpus)
        scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
        order = np.argsort(-scores)[:top_k]
        return [
            {"text": sentences[int(index)], "score": float(scores[index])}
            for index in order
            if scores[index] > 0
        ]
