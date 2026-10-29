# Context

## 2026-10-29 — Repo prep for first GitHub push

### Git hygiene
- `.gitignore` ignores generated/training output: `mlruns/`, `lightning_logs/`, `logs/`, `training_output/`, `optuna_studies/`, intermediate `*.ckpt`, `data/processed/*.parquet`, raw Rossmann CSVs, frontend coverage/test reports, tool caches, `.sisyphus/`.
- The only tracked model files are `checkpoints/tft/tft_latest.pt`, `checkpoints/tft/tft_latest.meta.json` and `checkpoints/rl/inventory_agent.zip`.
- Artifacts were removed from the index only (`git rm --cached`); they remain on disk. The staged size went from ~1.26 GB to ~26 MB (272 files).
- Deleted `docs/DEPLOYMENT.md` and `notebooks/03_CL_Experiment.ipynb` were dropped from the index; current `outputs/paper_figures/*.png` are staged.
- Branch renamed `master` → `main` (CI triggers on `main`/`develop`).

### Code / config
- Lint: ruff `--fix` + black applied across `src/`, `tests/`, `scripts/`; 22 long lines were wrapped by hand.
- `pyproject.toml`: ruff `select` moved to `[tool.ruff.lint]`; E402 is ignored for `scripts/*` (sys.path bootstrap) and `tests/*` (`pytest.importorskip` before imports).
- `src/api/routers/simulation.py`: removed the unused fill-rate accumulation (it was never returned).
- `tests/test_api/test_forecast_router.py`: the 503 assertion now matches the router's actual message ("Forecast model not loaded"); it was previously masked because the parquet was committed. The warmup-cache test now patches `get_tft_model` (the stale `_require_loaded_tft` no longer exists).
- `tests/test_api/test_inventory_simulation_routers.py`: removed `test_simulation_uses_public_observation_getter`. The router no longer steps an RL env (it uses a base-stock heuristic), so the test patched a non-existent `get_rl_agent`.
- `tests/test_integration.py`: skips when the raw Rossmann CSVs are absent (they are no longer committed).
- `requirements.txt` / `requirements.lock.txt`: added PyYAML, matplotlib, requests, tqdm.

### Docs
- README rewritten from the code. It has no deployment/Docker/license sections (per the user), and documents generating `tft_reference.parquet` via `recover_tft_artifacts()`.

### Verified on a clean export of the index (`git checkout-index`, .venv39 / Python 3.9)
- ruff, black, smoke test: pass. pytest: 148 passed, 4 skipped. Frontend: build OK, Vitest 76/76.
- **CI blocker:** `--cov-fail-under=80` fails at 63% total coverage (64% before these changes, so this predates them). Lowest: `data/experience_builder` 18%, `cl/strategy` 32%, `api/model_registry` 48%.

### Known, not addressed
- `docker/docker-compose.yml` build context (`.` = `docker/`) does not match `Dockerfile.api` COPY paths. The user asked not to cover deployment.
- The local venv `.venv39` is Python 3.9; CI uses 3.11.
- The user will add an MIT `LICENSE` file.
