from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel


router = APIRouter()


class PredictionRequest(BaseModel):
    subject: str = ""
    body: str
    sender: str = ""


class BatchPredictionRequest(BaseModel):
    items: list[PredictionRequest]


def _score_message(subject: str, body: str, sender: str) -> dict:
    text = f"{subject} {body}".lower()
    score = min(
        text.count("verify") * 0.15
        + text.count("password") * 0.2
        + text.count("urgent") * 0.1
        + (0.1 if "bank" in sender.lower() else 0.0),
        1.0,
    )
    return {"label": "phishing" if score >= 0.5 else "legitimate", "confidence": round(score, 4)}


@router.post("")
async def predict_single(payload: PredictionRequest) -> dict:
    return _score_message(payload.subject, payload.body, payload.sender)


@router.post("/batch")
async def predict_batch(payload: BatchPredictionRequest) -> dict:
    return {"count": len(payload.items), "results": [_score_message(item.subject, item.body, item.sender) for item in payload.items]}
