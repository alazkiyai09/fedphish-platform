# fedphish-platform

Production-oriented federated phishing detection platform. This split repo combines the FedPhish core system, vertical federated components, benchmark tooling, adversarial security experiments, dashboard assets, and the paper bundle.

## Layout

- `src/fedphish/`: preserved production FL system
- `src/federation/vertical/`: public vertical FL surfaces plus preserved legacy tree
- `src/security/`: public attack/defense/coevolution surfaces plus preserved legacy tree
- `src/benchmark/`: benchmark runner and preserved benchmark tree
- `src/api/`: unified FastAPI shell
- `src/dashboard/`: dashboard backend and frontend assets
- `paper/`: paper sources, figures, tables, and generation scripts

## Run

```bash
uvicorn src.api.main:app --reload
```

## Notes

- The repo keeps the richer original project trees while exposing a cleaner top-level structure.
- `tests/test_api.py` is the smoke test for the unified platform API shell.
