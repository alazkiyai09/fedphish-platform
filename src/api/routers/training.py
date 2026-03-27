from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel


router = APIRouter()


class TrainingRequest(BaseModel):
    clients: int = 5
    rounds: int = 10
    strategy: str = "FedAvg"


@router.post("/start")
async def start_training(payload: TrainingRequest, request: Request) -> dict:
    request.app.state.training = {
        "active": True,
        "round": 0,
        "config": payload.model_dump(),
    }
    return {"status": "started", **request.app.state.training}


@router.get("/status")
async def get_status(request: Request) -> dict:
    training = request.app.state.training
    if training["active"]:
        training["round"] = min(training["round"] + 1, training["config"]["rounds"])
    return training


@router.post("/stop")
async def stop_training(request: Request) -> dict:
    request.app.state.training["active"] = False
    return request.app.state.training
