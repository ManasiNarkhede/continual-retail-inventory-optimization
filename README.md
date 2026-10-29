# Retail Inventory Optimization

Demand forecasting and inventory control for retail stores, built on the Rossmann Store Sales dataset.

The pipeline combines three models:

1. **Temporal Fusion Transformer (TFT)**: multi-horizon, quantile demand forecasts (`pytorch-forecasting`).
2. **Continual learning (CL)**: adapts the forecaster to new groups of stores with Avalanche strategies (`Naive`, `EWC`, `Replay`) and an optional PackNet pruning scheduler.
3. **Reinforcement learning (RL)**: a PPO or SAC agent (Stable-Baselines3) that decides order quantities in a custom Gymnasium inventory environment.

A FastAPI backend serves forecasts, inventory actions, simulations, metrics, and alerts. A React + Vite dashboard reads from it.

---

## Architecture

```
data/raw/*.csv (Rossmann)
      │
      ▼
src/data/      load → validate → preprocess → feature engineering
      │
      ▼
src/models/    TFT training (PyTorch Lightning, MLflow logging)
      │
      ▼
src/cl/        continual learning across store "experiences" (Avalanche)
      │
      ▼
src/rl/        PPO / SAC inventory agent using forecasts as state
      │
      ▼
src/api/       FastAPI: /api/v1/{forecast, inventory, simulate, metrics, alerts}
      │
      ▼
frontend/      React dashboard (Vite, Tailwind, Recharts, React Query)
```

---

## Tech stack

| Area | Libraries |
|------|-----------|
| Forecasting | PyTorch 2.3, PyTorch Lightning 2.2, pytorch-forecasting 1.0 |
| Continual learning | avalanche-lib 0.5 |
| Reinforcement learning | Stable-Baselines3 2.3, Gymnasium 0.29 |
| Explainability | SHAP |
| HPO | Optuna |
| Tracking | MLflow |
| API | FastAPI, Uvicorn, Pydantic v2, SlowAPI (rate limiting) |
| Frontend | React 18, Vite 5, Tailwind CSS, Recharts, Zustand, TanStack Query, Framer Motion |
| Testing | pytest, Hypothesis, Vitest, Testing Library, jest-axe, Playwright |
| Tooling | ruff, black, mypy, GitHub Actions |

---

## Project structure

```
retail_inventory_opt/
├── .github/workflows/ci.yml   # Backend lint/test/smoke + frontend build/test
├── checkpoints/
│   ├── tft/tft_latest.pt      # Serving TFT weights (+ tft_latest.meta.json)
│   └── rl/inventory_agent.zip # Serving RL policy
├── configs/                   # YAML configs + JSON schemas + validator.py
├── data/
│   ├── raw/                   # Rossmann CSVs (not committed; see "Data")
│   └── processed/             # Runtime config, fingerprints, metrics JSON (+ generated parquet)
├── frontend/                  # React + Vite dashboard
├── notebooks/                 # EDA, TFT baseline, CL, RL notebooks
├── outputs/paper_figures/     # Generated figures
├── scripts/                   # Training, evaluation and diagnostic scripts
├── src/                       # Application source (see below)
└── tests/                     # pytest suites, mirroring src/
```

The following are generated locally and are **not committed** (see `.gitignore`): `data/processed/tft_reference.parquet`, `mlruns/`, `lightning_logs/`, `logs/`, `training_output/`, `optuna_studies/`, and intermediate `*.ckpt` checkpoints.

### Source modules (`src/`)

| Module | Contents |
|--------|----------|
| `api/main.py` | FastAPI app: CORS, request-id middleware, rate limiter, router registration, `/health` |
| `api/model_registry.py` | Singleton registry that loads the TFT, RL agent, and reference data, and recovers missing artifacts |
| `api/routers/` | `forecast`, `inventory`, `simulation`, `metrics`, `alerts` |
| `api/schemas.py` | Pydantic request/response models |
| `main.py` | Alternative app entrypoint (`src.main:app`) with config/env checks and optional auto-recovery training |
| `data/` | `loader`, `validator`, `schemas`, `preprocessor`, `feature_engineer`, `experience_builder` (Avalanche experiences) |
| `models/` | `tft_dataset`, `tft_model`, `tft_trainer`, `tft_interpretability`, `lr_finder` |
| `cl/` | `strategy` (Avalanche wrapper), `packnet`, `metrics` (BWT/FWT/plasticity/stability), `plugins`, `ablation` |
| `rl/` | `inventory_env`, `multi_store_env`, `reward` (triple-tier reward), `agent` (PPO/SAC), `curriculum`, `callbacks` |
| `eval/` | Forecast and inventory metrics, baselines, backtest harness, demand scenarios, reward sensitivity, SHAP explainer |
| `hpo/` | Optuna studies: `run_tft_hpo`, `run_rl_hpo` |
| `orchestration/` | Optional Prefect flow wrapping the full pipeline |
| `utils/` | Device selection (CPU/CUDA/MPS), JSON logging |

---

## Getting started

### Prerequisites

- Python 3.11 (used in CI; `pyproject.toml` requires ≥ 3.10)
- Node.js 20 (used in CI; 18+ is expected to work)
- Optional: a CUDA GPU for training

### Install

```bash
git clone <your-repo-url>
cd retail_inventory_opt

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install prefect locust        # optional: Prefect flow and Locust load test

cd frontend && npm install && cd ..
cp .env.example .env
cp frontend/.env.example frontend/.env
```

### Data

Download the [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales/data) files from Kaggle (you need an account and must accept the competition rules), and place them in `data/raw/`:

```
data/raw/train.csv
data/raw/test.csv
data/raw/store.csv
```

### Generate the reference dataset

The API serves forecasts from the committed checkpoints plus a feature-engineered reference file, `data/processed/tft_reference.parquet`. That file is not committed. Build it from the raw CSVs:

```bash
python -c "from src.api.model_registry import recover_tft_artifacts; recover_tft_artifacts()"
```

This runs the loader → preprocessor → feature-engineering pipeline and writes the parquet. If the file is missing and the raw CSVs are present, the API also rebuilds it automatically the first time it loads the TFT model. `scripts/train_tft.py` writes a fresh copy as well, along with a newly trained model.

### Environment variables

`.env.example`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server (use `file:./mlruns` without a server) |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | Bind address |
| `ENV` | `development` | `production` enables the startup checks in `src/main.py` |

`frontend/.env.example`: `VITE_API_TARGET` is the backend URL that the Vite dev server proxies `/api` to.

### Run locally

```bash
# Backend
uvicorn src.api.main:app --reload --port 8000

# Frontend
cd frontend
npm run dev        # http://localhost:5173
```

Interactive API docs are served at `http://localhost:8000/docs`.

---

## Training

```bash
# TFT forecaster (writes checkpoints/tft/ and data/processed/ artifacts)
python scripts/train_tft.py --config configs/tft_config.yaml
#   --store-id N            train on one store
#   --max-rows-per-store N  subsample for quick runs
#   --override-max-epochs N, --override-batch-size N, --override-hidden-size N, ...

# Continual learning (strategy set in configs/cl_config.yaml: Naive | EWC | Replay)
python scripts/train_cl.py --cl-config configs/cl_config.yaml

# RL agent (algorithm set in configs/rl_config.yaml: PPO | SAC)
python scripts/train_rl.py --config configs/rl_config.yaml --total-timesteps 500000
```

Other useful scripts:

| Script | Purpose |
|--------|---------|
| `scripts/smoke_test.py` | Import smoke test for all core modules (run in CI) |
| `scripts/smoke_train_tft.py` | Short TFT training pass on real data |
| `scripts/run_cl_ablation.py` | Compare CL strategies |
| `scripts/run_reward_sensitivity.py` | RL reward sensitivity / Pareto sweep |
| `scripts/generate_paper_figures.py` | Regenerate `outputs/paper_figures/` |
| `scripts/demo_runner.py [--full]` | End-to-end demo |
| `scripts/load_test.py --base-url URL --requests N` | API load test (`--use-testclient` for in-process) |

Hyperparameter search runs from Python:

```python
from src.hpo.tft_hpo import run_tft_hpo
from src.hpo.rl_hpo import run_rl_hpo
```

Track experiments with `mlflow ui --backend-store-uri mlruns/`.

---

## API

Every router is mounted under both `/api/v1` and `/api`.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Status (`ok` / `degraded`), version, model-registry state |
| POST | `/api/v1/forecast/` | Quantile demand forecast for a store |
| POST | `/api/v1/forecast/batch` | Forecasts for several stores |
| POST | `/api/v1/inventory/action` | RL-recommended order quantity |
| POST | `/api/v1/simulate/` | Inventory rollout trajectory |
| GET | `/api/v1/metrics/kpis` | Dashboard KPIs |
| GET | `/api/v1/metrics/cl` | Continual-learning metrics |
| GET | `/api/v1/metrics/stores` | Available stores |
| GET | `/api/v1/metrics/simulate/single` | Single-store simulation summary |
| GET | `/api/v1/metrics/simulate/batch` | Multi-store simulation summary |
| GET | `/api/v1/alerts/active` | Active stock / model-health alerts |
| POST | `/api/v1/alerts/check/{store_id}` | Evaluate alert rules for a store |
| GET | `/api/v1/alerts/rules` | Configured alert rules |

---

## Dashboard

Routes defined in `frontend/src/App.jsx`:

| Route | Page |
|-------|------|
| `/` | Home |
| `/dashboard` | KPI overview and forecast chart |
| `/forecast` | Per-store demand forecast workspace |
| `/simulation` | Inventory simulation per store (horizon in days, risk-aversion setting) |
| `/cl-metrics` | Continual-learning curves and metrics |
| `/hpo` | Hyperparameter optimization view |

```bash
cd frontend
npm run build          # build → frontend/dist/
npm run test           # Vitest unit tests
npm run test:coverage
npm run test:e2e       # Playwright
```

---

## Results

These are the values stored in the repository's metrics files. This README has not re-run them.

**Forecasting and inventory** (`data/processed/metrics_latest.json`, 1,115 stores):

| Metric | TFT | Baseline |
|--------|-----|----------|
| Validation sMAPE | 7.78 % | 20.48 % |
| Validation RMSE | 862.26 | 1303.86 |
| p10–p90 interval coverage | 80.4 % | – |
| Service level | 96.7 % | – |
| Fill rate | 94.3 % | – |

**Continual learning** (`data/processed/cl_metrics_latest.json`, Replay strategy; the file's `source` field is `offline_tuned_profile`):

| Metric | Value | Target (`configs/cl_config.yaml`) |
|--------|-------|-----------------------------------|
| Backward transfer (BWT) | −0.0185 | ≥ −0.02 |
| Forward transfer (FWT) | 0.0241 | ≥ 0.01 |
| Stability | 0.912 | ≥ 0.90 |
| Plasticity | 0.867 | ≥ 0.86 |

---

## Testing

```bash
pytest                                   # all tests (config in pyproject.toml)
pytest --cov=src --cov-report=html       # coverage (fail_under = 80)
ruff check src/ tests/ scripts/
black --check src/ tests/ scripts/
mypy                                     # strict on the files listed in pyproject.toml
python scripts/smoke_test.py
```

CI (`.github/workflows/ci.yml`) runs ruff, black, pytest with coverage, and the smoke test on Python 3.11, then builds and tests the frontend on Node 20.

---

## Acknowledgements

- [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) dataset (Kaggle)
- PyTorch Forecasting, Avalanche, Stable-Baselines3, FastAPI, MLflow, Optuna
