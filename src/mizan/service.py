"""Inference orchestration for evidence-backed verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .model import MizanClassifier
from .retriever import EvidenceRetriever


@dataclass
class VerifierService:
    classifier: MizanClassifier
    retriever: EvidenceRetriever
    min_retrieval_score: float = 0.05
    min_confidence: float = 0.45

    def metadata(self) -> dict[str, Any]:
        return {
            **self.classifier.metadata(),
            "retrieval_type": "TF-IDF cosine similarity over verification articles",
            "min_retrieval_score": self.min_retrieval_score,
            "min_confidence": self.min_confidence,
            "safety_policy": "insufficient_evidence when retrieval or confidence is below threshold",
        }

    def verify(self, claim: str, top_k: int = 3) -> dict[str, Any]:
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError("claim must be a non-empty string")
        candidates = self.retriever.retrieve(claim, top_k=top_k)
        if not candidates:
            return {
                "claim": claim,
                "verdict": None,
                "confidence": 0.0,
                "evidence_status": "insufficient_evidence",
                "candidates": [],
                "model_version": self.classifier.version,
            }

        top = candidates[0]
        prediction_frame = pd.DataFrame([{"claim": claim, "content": top["content"]}])
        verdict = self.classifier.predict(prediction_frame)[0]
        probabilities = self.classifier.predict_proba(prediction_frame)[0]
        confidence = max(probabilities.values())
        evidence_sentences = self.retriever.rank_sentences(claim, top["content"], top_k=2)
        has_evidence = top["score"] >= self.min_retrieval_score and bool(evidence_sentences)
        safe_to_decide = has_evidence and confidence >= self.min_confidence

        return {
            "claim": claim,
            "verdict": verdict if safe_to_decide else None,
            "model_verdict": verdict,
            "confidence": round(float(confidence), 6),
            "evidence_status": "sufficient" if safe_to_decide else "insufficient_evidence",
            "retrieval_score": round(float(top["score"]), 6),
            "probabilities": {key: round(float(value), 6) for key, value in probabilities.items()},
            "evidence": {
                "claim_id": top["claim_id"],
                "source": top["source"],
                "date": top["date"],
                "source_url": top["source_url"],
                "sentences": evidence_sentences,
            },
            "candidates": [
                {
                    "claim_id": item["claim_id"],
                    "source": item["source"],
                    "date": item["date"],
                    "label": item["label"],
                    "score": round(float(item["score"]), 6),
                    "source_url": item["source_url"],
                }
                for item in candidates
            ],
            "model_version": self.classifier.version,
        }
