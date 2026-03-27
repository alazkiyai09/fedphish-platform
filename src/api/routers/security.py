from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel


router = APIRouter()


class CoevolutionRequest(BaseModel):
    attack: str = "label_flip"
    defense: str = "anomaly_detection"
    rounds: int = 5


@router.post("/coevolution/run")
async def run_coevolution(payload: CoevolutionRequest, request: Request) -> dict:
    run_id = str(uuid4())
    result = {
        "run_id": run_id,
        "attack": payload.attack,
        "defense": payload.defense,
        "rounds": payload.rounds,
        "winner": "defense" if payload.defense != "none" else "attack",
    }
    request.app.state.coevolution_runs[run_id] = result
    return result


@router.get("/coevolution/{run_id}")
async def get_coevolution(run_id: str, request: Request) -> dict:
    result = request.app.state.coevolution_runs.get(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return result


@router.get("/game-theory")
async def get_game_theory() -> dict:
    return {"nash_equilibrium": {"attack": "gradual_backdoor", "defense": "honeypot"}, "payoff": 0.73}
