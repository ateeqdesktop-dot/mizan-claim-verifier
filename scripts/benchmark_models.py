#!/usr/bin/env python3
"""Benchmark simple claim-only baselines for Mizan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from mizan.data import LABELS, chronological_split, load_arafacts, stratified_split
from mizan.model import MizanClassifier
from mizan.text import normalize_arabic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", required=True, type=Path)
    parser.add_argument("--content", required=True, type=Path)
    parser.add_argument("--output", default=Path("artifacts/model_benchmark.json"), type=Path)
    return parser.parse_args()


def feature_pipeline(estimator) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=65000),
            ),
            (
                "char",
                TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=90000),
            ),
        ]
    )
    return Pipeline([("features", features), ("classifier", estimator)])


def score(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    return {
        "rows": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
    }


def evaluate(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, dict[str, float]]:
    x_train = [normalize_arabic(value) for value in train["claim"]]
    x_test = [normalize_arabic(value) for value in test["claim"]]
    y_train = train["normalized_label"].astype(str).tolist()
    y_test = test["normalized_label"].astype(str).tolist()

    models = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "linear_svc_balanced": feature_pipeline(LinearSVC(class_weight="balanced", C=1.0, random_state=42)),
    }
    results: dict[str, dict[str, float]] = {}
    for name, estimator in models.items():
        estimator.fit(x_train, y_train)
        results[name] = score(y_test, [str(value) for value in estimator.predict(x_test)])

    logistic = MizanClassifier(include_evidence=False).fit(train)
    results["logistic_balanced_word_char"] = score(y_test, logistic.predict(test))
    return results


def main() -> None:
    args = parse_args()
    frame = load_arafacts(args.claims, args.content)
    chronological = chronological_split(frame)
    stratified_train, stratified_test = stratified_split(frame, test_size=0.20, random_state=42)
    output = {
        "task": "claim-only classification",
        "labels": LABELS,
        "protocols": {
            "chronological": evaluate(chronological.train, chronological.test),
            "stratified": evaluate(stratified_train, stratified_test),
        },
        "selection_note": "The deployable model remains logistic_balanced_word_char because it provides probability estimates used by the abstention policy; these probabilities are not independently calibrated in this release, and benchmark scores alone do not justify replacing the serving model.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
