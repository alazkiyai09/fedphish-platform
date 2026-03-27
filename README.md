<div align="center">

# 🛰️ FedPhish Platform

### Federated Phishing Detection • Coevolution Security • Benchmarking • Dashboard

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Federated Learning](https://img.shields.io/badge/Federated-Learning-326CE5?style=flat)](https://flower.ai/)
[![WebSocket](https://img.shields.io/badge/WebSocket-Real--Time-4CAF50?style=flat)](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API)

[Overview](#-overview) • [About](#-about) • [Topics](#-topics) • [API](#-api-surfaces) • [Quick Start](#-quick-start)

---

End-to-end platform for **federated phishing model training**, **security attack-defense coevolution**, **benchmark automation**, and **dashboard-assisted operations**.

</div>

---

## 🎯 Overview

`fedphish-platform` unifies:

- Federated phishing prediction and training workflows
- Security coevolution simulation (attacker vs defender)
- Benchmark orchestration and scenario management
- Dashboard-ready API and WebSocket streams

## 📌 About

- Built to connect research-grade FL security with platform operations
- Consolidates APIs, simulators, configs, and benchmark assets
- Suitable for iterative red-team/blue-team evaluation cycles

## 🏷️ Topics

`fedphish` `federated-learning` `phishing-detection` `adversarial-ml` `security-simulation` `fastapi` `websocket` `benchmarking`

## 🧩 Architecture

- `src/fedphish/`: core platform services
- `src/federation/vertical/`: vertical FL workflows
- `src/security/`: attacks, defenses, coevolution logic
- `src/benchmark/`: benchmark pipelines and configs
- `src/dashboard/`: backend/frontend dashboard modules
- `src/api/`: unified API entrypoint

## 🌐 API Surfaces

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
- `WS /ws/simulation`

## ⚡ Quick Start

```bash
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

## 🛠️ Tech Stack

**Core:** FastAPI, Pydantic, WebSockets  
**FL/Security:** federated training + adversarial simulation modules  
**Ops:** dashboard backend, benchmark config assets
