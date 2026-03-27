# FedPhish Platform (`fedphish-platform`)

Full-stack **federated phishing detection platform** that combines model training, vertical federation workflows, adversarial security simulation, benchmark orchestration, and dashboard operations.

## Why This Repository

Most FL security demos split training, security, and visualization into disconnected projects. `fedphish-platform` integrates these surfaces into one platform for research-to-production handoff.

## Core Features

- Federated phishing training modules (client/server/privacy/detection)
- Vertical FL and PSI integration surfaces
- Security coevolution utilities for attack-vs-defense simulation
- Benchmark runner and scenario config structure
- Unified FastAPI endpoints for prediction, training, security, benchmark
- Dashboard backend/frontend structure for demo and monitoring workflows

## Project Structure

- `src/fedphish/`: core platform modules (client/server/privacy/security)
- `src/federation/vertical/`: vertical FL workflows and interfaces
- `src/security/`: attacks, defenses, coevolution, game-theory surfaces
- `src/benchmark/`: benchmark runner and config assets
- `src/api/`: unified platform FastAPI app
- `src/dashboard/`: backend + frontend dashboard components

## API Endpoints

- `POST /api/v1/predict`
- `POST /api/v1/predict/batch`
- `POST /api/v1/training/start`
- `GET /api/v1/training/status`
- `POST /api/v1/training/stop`
- `POST /api/v1/benchmark/run`
- `GET /api/v1/benchmark/results`
- `POST /api/v1/security/coevolution/run`
- `GET /api/v1/security/coevolution/{run_id}`
- `GET /api/v1/security/game-theory`
- `GET /health`
- `GET /metrics`

## Quick Start

```bash
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

## SEO Keywords

federated phishing detection, fedphish platform, vertical federated learning phishing, adversarial federated learning security, coevolution attack defense simulation, fastapi federated ai platform
