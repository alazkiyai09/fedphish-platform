"""
FedPhish Demo Dashboard - Main FastAPI Application.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import AppConfig, SCENARIOS
from .websocket.handlers import handler
from .websocket.manager import manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load configuration
config = AppConfig()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting FedPhish Demo Dashboard backend")
    yield
    logger.info("Shutting down FedPhish Demo Dashboard backend")


# Create FastAPI app
app = FastAPI(
    title="FedPhish Demo Dashboard",
    description="Backend API for federated phishing detection demo",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REST API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "FedPhish Demo Dashboard Backend",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "scenarios": "/api/v1/scenarios",
            "websocket": "/ws/federation",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/api/v1/scenarios")
async def list_scenarios():
    """List available demo scenarios."""
    return {
        "scenarios": [
            {
                "name": name,
                "description": scenario.description,
                "num_banks": scenario.num_banks,
                "num_rounds": scenario.num_rounds,
                "privacy_level": scenario.privacy_level,
            }
            for name, scenario in SCENARIOS.items()
        ]
    }


@app.post("/api/v1/scenarios/{scenario_name}")
async def load_scenario(scenario_name: str):
    """Load a specific scenario."""
    if scenario_name not in SCENARIOS:
        raise HTTPException(status_code=404, detail="Scenario not found")

    scenario = SCENARIOS[scenario_name]

    # Create simulator
    from .core.simulator import FederatedSimulator
    from .websocket.handlers import handler as ws_handler

    if scenario_name not in ws_handler.simulators:
        ws_handler.simulators[scenario_name] = FederatedSimulator(scenario)

    return {
        "scenario": scenario_name,
        "config": scenario.dict(),
        "status": "loaded",
    }


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws/federation")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()

            # Handle message
            await handler.handle_message(websocket, data)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        reload=True,
        log_level="info",
    )
