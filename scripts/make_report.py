#!/usr/bin/env python3
"""Create compact evaluation figures from Mizan artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

LABELS = ["False", "Partly-false", "True", "Sarcasm", "Unverifiable"]


def plot_confusion(path: Path, output: Path, title: str) -> None:
    matrix = pd.read_csv(path, index_col=0)
    matrix = matrix.reindex(index=LABELS, columns=LABELS, fill_value=0)
    plt.figure(figsize=(7.2, 5.8))
    sns.heatmap(matrix, annot=True, fmt="g", cmap="Blues", cbar=False, xticklabels=LABELS, yticklabels=LABELS)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    with (args.artifacts / "dataset_summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)

    counts = pd.Series(summary["label_counts"]).reindex(LABELS).fillna(0)
    plt.figure(figsize=(8.5, 4.8))
    ax = counts.plot(kind="bar", color="#2f6690")
    ax.set_title("AraFacts label distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Rows")
    ax.tick_params(axis="x", rotation=25)
    for index, value in enumerate(counts):
        ax.text(index, value + max(counts) * 0.015, f"{int(value):,}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(args.artifacts / "label_distribution.png", dpi=180)
    plt.close()

    plot_confusion(
        args.artifacts / "chronological_claim_only_metrics_confusion_matrix.csv",
        args.artifacts / "chronological_claim_only_confusion.png",
        "Chronological test: claim-only model",
    )
    plot_confusion(
        args.artifacts / "stratified_claim_only_metrics_confusion_matrix.csv",
        args.artifacts / "stratified_claim_only_confusion.png",
        "Stratified test: claim-only model",
    )
    print("created evaluation figures")


if __name__ == "__main__":
    main()
