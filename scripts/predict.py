#!/usr/bin/env python3
"""Run one evidence-backed prediction from persisted Mizan artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd

from mizan.model import MizanClassifier
from mizan.retriever import EvidenceRetriever
from mizan.service import VerifierService


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("claim", help="Arabic claim to verify")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    classifier = MizanClassifier.load(args.model_dir / "classifier.joblib")
    retriever = joblib.load(args.model_dir / "retriever.joblib")
    if not isinstance(retriever, EvidenceRetriever):
        raise TypeError("retriever artifact has an unexpected type")
    result = VerifierService(classifier, retriever).verify(args.claim, top_k=args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
