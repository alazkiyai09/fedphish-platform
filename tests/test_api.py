from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_training_lifecycle() -> None:
    start = client.post("/api/v1/training/start", json={"clients": 3, "rounds": 4, "strategy": "FedAvg"})
    assert start.status_code == 200
    status = client.get("/api/v1/training/status")
    assert status.status_code == 200
    stop = client.post("/api/v1/training/stop")
    assert stop.status_code == 200


def test_security_and_predict() -> None:
    predict = client.post(
        "/api/v1/predict",
        json={"subject": "Verify your account", "body": "Urgent password reset required", "sender": "alerts@bank.example"},
    )
    assert predict.status_code == 200
    security = client.post("/api/v1/security/coevolution/run", json={"attack": "label_flip", "defense": "honeypot", "rounds": 3})
    assert security.status_code == 200
