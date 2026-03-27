from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Request
from pydantic import BaseModel


router = APIRouter()


class BenchmarkRequest(BaseModel):
    scenario: str = "standard"
    strategies: list[str] = ["FedAvg", "FedProx"]


@router.post("/run")
async def run_benchmark(payload: BenchmarkRequest, request: Request) -> dict:
    result = {
        "benchmark_id": f"bench-{len(request.app.state.benchmarks) + 1}",
        "created_at": datetime.now(UTC).isoformat(),
        "scenario": payload.scenario,
        "strategies": payload.strategies,
        "summary": {"best_strategy": payload.strategies[0], "f1": 0.94, "robustness": 0.89},
    }
    request.app.state.benchmarks.append(result)
    return result


@router.get("/results")
async def get_results(request: Request) -> dict:
    return {"count": len(request.app.state.benchmarks), "items": request.app.state.benchmarks}
