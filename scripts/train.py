#!/usr/bin/env python3
"""Train, evaluate, and persist the Mizan model on AraFacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from mizan.data import LABELS, chronological_split, load_arafacts, stratified_split
from mizan.model import MizanClassifier
from mizan.retriever import EvidenceRetriever
from mizan.service import VerifierService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", required=True, type=Path)
    parser.add_argument("--content", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("artifacts"), type=Path)
    parser.add_argument("--model-dir", default=Path("models"), type=Path)
    return parser.parse_args()


def metrics_for(y_true: list[str], y_pred: list[str]) -> dict:
    return {
        "rows": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def evaluate_split(train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    y_true = test["normalized_label"].astype(str).tolist()
    claim_classifier = MizanClassifier(include_evidence=False).fit(train)
    evidence_classifier = MizanClassifier(include_evidence=True).fit(train)
    retriever = EvidenceRetriever().fit(train)

    claim_pred = claim_classifier.predict(test)
    oracle_pred = evidence_classifier.predict(test)

    retrieved_rows: list[dict] = []
    top1_label_matches = 0
    retrieval_scores: list[float] = []
    for _, row in test.iterrows():
        candidates = retriever.retrieve(str(row["claim"]), top_k=3)
        if candidates:
            top = candidates[0]
            top1_label_matches += int(top["label"] == row["normalized_label"])
            retrieval_scores.append(float(top["score"]))
            retrieved_rows.append({"claim": row["claim"], "content": top["content"]})
        else:
            retrieved_rows.append({"claim": row["claim"], "content": ""})
    retrieved_frame = pd.DataFrame(retrieved_rows)
    retrieved_pred = evidence_classifier.predict(retrieved_frame)

    evaluation = {
        "claim_only_metrics": metrics_for(y_true, claim_pred),
        "oracle_evidence_metrics": metrics_for(y_true, oracle_pred),
        "retrieved_evidence_metrics": metrics_for(y_true, retrieved_pred),
        "retrieval_metrics": {
            "top1_retrieved_label_match_rate": float(top1_label_matches / max(1, len(test))),
            "mean_top1_retrieval_score": float(np.mean(retrieval_scores)) if retrieval_scores else 0.0,
        },
    }
    predictions = test.assign(
        claim_only_prediction=claim_pred,
        oracle_evidence_prediction=oracle_pred,
        retrieved_evidence_prediction=retrieved_pred,
    )
    return evaluation, predictions


def save_confusion_matrices(evaluation: dict, output_dir: Path, prefix: str) -> None:
    for name in ["claim_only_metrics", "oracle_evidence_metrics", "retrieved_evidence_metrics"]:
        pd.DataFrame(evaluation[name]["confusion_matrix"], index=LABELS, columns=LABELS).to_csv(
            output_dir / f"{prefix}_{name}_confusion_matrix.csv"
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    frame = load_arafacts(args.claims, args.content)
    chronological = chronological_split(frame)
    chronological_metrics, chronological_predictions = evaluate_split(chronological.train, chronological.test)

    strat_train, strat_test = stratified_split(frame, test_size=0.20, random_state=42)
    stratified_metrics, stratified_predictions = evaluate_split(strat_train, strat_test)

    # The deployable classifier is claim-only: evidence is retrieved separately and shown for review.
    final_classifier = MizanClassifier(include_evidence=False).fit(frame)
    final_retriever = EvidenceRetriever().fit(frame)
    final_classifier.save(args.model_dir / "classifier.joblib")
    joblib.dump(final_retriever, args.model_dir / "retriever.joblib")

    cols = ["ClaimID", "claim", "normalized_label", "source", "date", "claim_only_prediction", "oracle_evidence_prediction", "retrieved_evidence_prediction"]
    chronological_predictions[cols].to_csv(args.output_dir / "chronological_test_predictions.csv", index=False)
    stratified_predictions[cols].to_csv(args.output_dir / "stratified_test_predictions.csv", index=False)
    save_confusion_matrices(chronological_metrics, args.output_dir, "chronological")
    save_confusion_matrices(stratified_metrics, args.output_dir, "stratified")

    metrics = {
        "dataset_rows": len(frame),
        "chronological_split_rows": {"train": len(chronological.train), "validation": len(chronological.validation), "test": len(chronological.test)},
        "stratified_split_rows": {"train": len(strat_train), "test": len(strat_test)},
        "label_counts": frame["normalized_label"].value_counts().to_dict(),
        "chronological": {
            **chronological_metrics,
            "date_ranges": {
                "train": [str(chronological.train["date_parsed"].min()), str(chronological.train["date_parsed"].max())],
                "validation": [str(chronological.validation["date_parsed"].min()), str(chronological.validation["date_parsed"].max())],
                "test": [str(chronological.test["date_parsed"].min()), str(chronological.test["date_parsed"].max())],
            },
        },
        "stratified": stratified_metrics,
        "model_metadata": final_classifier.metadata(),
        "protocol": "chronological 70/15/15 plus stratified 80/20; retrieval is train-only during evaluation; final artifacts fit all historical rows",
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    with (args.output_dir / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "rows": len(frame),
                "columns": list(frame.columns),
                "label_counts": frame["normalized_label"].value_counts().to_dict(),
                "source_counts": frame["source"].value_counts().to_dict(),
                "date_min": str(frame["date_parsed"].min()),
                "date_max": str(frame["date_parsed"].max()),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    service = VerifierService(final_classifier, final_retriever)
    with (args.output_dir / "service_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(service.metadata(), handle, ensure_ascii=False, indent=2)

    print(json.dumps({"chronological": chronological_metrics, "stratified": stratified_metrics}, ensure_ascii=False, indent=2))
    print(f"saved model artifacts to {args.model_dir}")


if __name__ == "__main__":
    main()
