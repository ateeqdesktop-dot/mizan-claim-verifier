import pandas as pd
from fastapi.testclient import TestClient

from mizan.api import app, configure_service
from mizan.model import MizanClassifier
from mizan.retriever import EvidenceRetriever
from mizan.service import VerifierService


def make_training_frame() -> pd.DataFrame:
    rows = []
    examples = [
        ("خبر صحيح عن الطقس", "هذا المقال يثبت أن الخبر صحيح.", "True"),
        ("خبر صحيح عن التعليم", "هذا المقال يثبت أن الخبر صحيح.", "True"),
        ("شائعة كاذبة عن الطقس", "التحقق من شائعة كاذبة عن الطقس يثبت أنها زائفة.", "False"),
        ("شائعة كاذبة عن السفر", "التحقق من شائعة كاذبة عن السفر يثبت أنها زائفة.", "False"),
        ("تصريح مضلل عن الصحة", "التحقق يوضح أن التصريح صحيح جزئيًا.", "Partly-false"),
        ("تصريح مضلل عن الاقتصاد", "التحقق يوضح أن التصريح صحيح جزئيًا.", "Partly-false"),
    ]
    for index, (claim, content, label) in enumerate(examples):
        rows.append(
            {
                "ClaimID": f"T{index}",
                "claim": claim,
                "content": content,
                "content_normalized": content,
                "source": "test",
                "date": "2024-01-01",
                "normalized_label": label,
            }
        )
    return pd.DataFrame(rows)


def test_service_returns_reviewable_evidence():
    frame = make_training_frame()
    service = VerifierService(
        classifier=MizanClassifier(include_evidence=False).fit(frame),
        retriever=EvidenceRetriever().fit(frame),
        min_retrieval_score=0.0,
        min_confidence=0.0,
    )
    result = service.verify("شائعة كاذبة عن الطقس", top_k=2)
    assert result["evidence_status"] == "sufficient"
    assert result["evidence"]["sentences"]
    assert len(result["candidates"]) == 2


def test_api_health_demo_and_metadata():
    frame = make_training_frame()
    service = VerifierService(
        classifier=MizanClassifier(include_evidence=False).fit(frame),
        retriever=EvidenceRetriever().fit(frame),
        min_retrieval_score=0.0,
        min_confidence=0.0,
    )
    configure_service(service)
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/demo").status_code == 200
    response = client.post("/verify", json={"claim": "خبر صحيح عن الطقس", "top_k": 1})
    assert response.status_code == 200
    payload = response.json()
    assert "verdict" in payload and "evidence" in payload
    assert client.get("/metadata").status_code == 200


def test_api_rejects_invalid_claim():
    client = TestClient(app)
    response = client.post("/verify", json={"claim": "x", "top_k": 1})
    assert response.status_code == 422


def test_service_abstains_when_confidence_threshold_is_high():
    frame = make_training_frame()
    service = VerifierService(
        classifier=MizanClassifier(include_evidence=False).fit(frame),
        retriever=EvidenceRetriever().fit(frame),
        min_retrieval_score=0.0,
        min_confidence=1.1,
    )
    result = service.verify("شائعة كاذبة عن الطقس", top_k=1)
    assert result["verdict"] is None
    assert result["evidence_status"] == "insufficient_evidence"
    assert result["model_verdict"] in {"True", "False", "Partly-false"}
