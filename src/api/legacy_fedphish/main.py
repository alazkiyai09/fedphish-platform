"""
FedPhish FastAPI Application.

REST API for phishing detection and federated learning management.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="FedPhish API",
    description="Federated Phishing Detection for Financial Institutions",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Request model for prediction."""

    text: str = Field(..., description="Text to classify as phishing/legitimate")
    explain: bool = Field(False, description="Whether to return explanation")

    class Config:
        schema_extra = {
            "example": {
                "text": "Urgent: Your account will be suspended. Click here to verify.",
                "explain": True,
            }
        }


class PredictionResponse(BaseModel):
    """Response model for prediction."""

    prediction: str = Field(..., description="Prediction: 'phishing' or 'legitimate'")
    confidence: float = Field(..., description="Confidence score (0-1)")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    explanation: Optional[str] = Field(None, description="Explanation if requested")


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction."""

    texts: List[str] = Field(..., description="List of texts to classify")
    explain: bool = Field(False, description="Whether to return explanations")

    class Config:
        schema_extra = {
            "example": {
                "texts": [
                    "Urgent: Verify your account now",
                    "Your monthly statement is ready",
                ],
                "explain": False,
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction."""

    predictions: List[str] = Field(..., description="Predictions for each text")
    confidences: List[float] = Field(..., description="Confidence scores")
    probabilities: List[Dict[str, float]] = Field(..., description="Class probabilities")
    explanations: Optional[List[str]] = Field(None, description="Explanations if requested")


class TrainingStatusResponse(BaseModel):
    """Response model for training status."""

    status: str = Field(..., description="Training status")
    current_round: int = Field(..., description="Current round number")
    total_rounds: int = Field(..., description="Total rounds")
    accuracy: float = Field(..., description="Current accuracy")
    loss: float = Field(..., description="Current loss")


# Global model manager (singleton)
class ModelManager:
    """Manage model lifecycle."""

    def __init__(self):
        """Initialize model manager."""
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False

    def load_model(self, model_path: Optional[str] = None):
        """Load model."""
        if self.is_loaded:
            return

        try:
            from transformers import AutoTokenizer

            from src.fedphish.client.model import create_model

            logger.info(f"Loading model on device: {self.device}")

            # Create or load model
            if model_path and Path(model_path).exists():
                self.model = create_model(device=self.device)
                self.model.load(model_path)
            else:
                # Load pre-trained model
                self.model = create_model(
                    model_name="distilbert-base-uncased",
                    device=self.device,
                )

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

            self.model.eval_mode()
            self.is_loaded = True

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def predict(self, text: str) -> Dict[str, Any]:
        """Make prediction."""
        if not self.is_loaded:
            self.load_model()

        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Predict
            with torch.no_grad():
                predictions, probabilities = self.model.predict(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                )

            # Convert to labels
            label_map = {0: "legitimate", 1: "phishing"}
            prediction = label_map[predictions[0].item()]
            confidence = probabilities[0][predictions[0]].item()

            probs_dict = {
                "legitimate": probabilities[0][0].item(),
                "phishing": probabilities[0][1].item(),
            }

            return {
                "prediction": prediction,
                "confidence": confidence,
                "probabilities": probs_dict,
            }

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise


# Create model manager
model_manager = ModelManager()


# Health check endpoint
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "fedphish-api",
        "version": "0.1.0",
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "FedPhish API",
        "description": "Federated Phishing Detection for Financial Institutions",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "predict": "/api/v1/predict",
            "batch_predict": "/api/v1/predict/batch",
            "status": "/api/v1/training/status",
        },
    }


# Prediction endpoint
@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
)
async def predict(request: PredictionRequest):
    """
    Make phishing prediction.

    Args:
        request: Prediction request

    Returns:
        Prediction response
    """
    try:
        # Get prediction
        result = model_manager.predict(request.text)

        # Add explanation if requested
        explanation = None
        if request.explain:
            explanation = generate_explanation(request.text, result)

        return PredictionResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            explanation=explanation,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


# Batch prediction endpoint
@app.post(
    "/api/v1/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
)
async def batch_predict(request: BatchPredictionRequest):
    """
    Make batch phishing predictions.

    Args:
        request: Batch prediction request

    Returns:
        Batch prediction response
    """
    try:
        predictions = []
        confidences = []
        probabilities_list = []
        explanations = []

        for text in request.texts:
            result = model_manager.predict(text)

            predictions.append(result["prediction"])
            confidences.append(result["confidence"])
            probabilities_list.append(result["probabilities"])

            if request.explain:
                explanations.append(generate_explanation(text, result))

        return BatchPredictionResponse(
            predictions=predictions,
            confidences=confidences,
            probabilities=probabilities_list,
            explanations=explanations if request.explain else None,
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


# Training status endpoint
@app.get(
    "/api/v1/training/status",
    response_model=TrainingStatusResponse,
)
async def get_training_status():
    """Get federated training status.

    Returns current training state from in-memory tracker.
    In production, this would query a persistent state store.
    """
    from src.fedphish.api.training_state import get_training_state

    state = get_training_state()
    return TrainingStatusResponse(
        status=state.get("status", "idle"),
        current_round=state.get("current_round", 0),
        total_rounds=state.get("total_rounds", 20),
        accuracy=state.get("accuracy", 0.0),
        loss=state.get("loss", 0.0),
    )


# Utility functions
def generate_explanation(text: str, result: Dict[str, Any]) -> str:
    """Generate explanation for prediction."""
    pred = result["prediction"]
    conf = result["confidence"]

    if pred == "phishing":
        explanation = (
            f"This message is classified as PHISHING with {conf:.1%} confidence. "
            f"Key indicators: urgent language, request for sensitive information, "
            f"suspicious links or attachments. Recommended action: Do not click links "
            f"or download attachments. Report to security team."
        )
    else:
        explanation = (
            f"This message appears LEGITIMATE with {conf:.1%} confidence. "
            f"However, always verify sender identity and be cautious with "
            f"requests for sensitive information."
        )

    return explanation


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize API on startup."""
    logger.info("Starting FedPhish API")

    # Load model eagerly
    try:
        model_manager.load_model()
        logger.info("Model loaded successfully on startup")
    except Exception as e:
        logger.warning(f"Model not loaded on startup: {e}. Will load on first request.")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down FedPhish API")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
