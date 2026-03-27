from fastapi import FastAPI

from src.api.routers import benchmark, predict, security, training


app = FastAPI(
    title="fedphish-platform",
    version="0.1.0",
    description="Unified platform API for prediction, federated training, benchmarking, and security scenarios.",
)

app.include_router(predict.router, prefix="/api/v1/predict", tags=["predict"])
app.include_router(training.router, prefix="/api/v1/training", tags=["training"])
app.include_router(benchmark.router, prefix="/api/v1/benchmark", tags=["benchmark"])
app.include_router(security.router, prefix="/api/v1/security", tags=["security"])


@app.on_event("startup")
async def startup() -> None:
    app.state.training = {"active": False, "round": 0, "config": None}
    app.state.benchmarks = []
    app.state.coevolution_runs = {}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "fedphish-platform"}


@app.get("/metrics")
async def metrics() -> dict:
    return {
        "training_active": app.state.training["active"],
        "benchmark_count": len(app.state.benchmarks),
        "coevolution_runs": len(app.state.coevolution_runs),
    }
