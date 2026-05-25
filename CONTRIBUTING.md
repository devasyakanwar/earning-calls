# Contributing Guidelines

Welcome to the Multimodal Earnings Call Intelligence System. We follow a strict **Data Contract** approach to ensure parallel development.

## Project Structure
- `src/preprocessing`: Data ingestion, speaker classification, and DB management.
- `src/features`: Feature extraction pipelines (text, audio, interaction).
- `src/modeling`: Training and benchmarking (LightGBM, fusion network).
- `src/evaluation`: Metrics, leakage control, and backtesting.
- `src/dashboard`: Streamlit analyst dashboard.
- `src/inference`: Production inference pipeline.

## Development Workflow
1. **Branching**: Use feature branches (`feat/audio-extraction`).
2. **Data Contracts**: All new features must be saved as Parquet files in `data/processed/` and adhere to the agreed-upon schema in the implementation plan.
3. **Tests**: Add unit tests in `tests/` for any new extraction logic.
4. **Dependencies**: Add all dependencies to `pyproject.toml` (canonical source), then sync `requirements.txt`.

## Coding Standards
- Use type hints wherever possible.
- Use `polars` for data manipulation to ensure scalability.
- Document all extractors with docstrings explaining the features.
- Only call `logging.basicConfig()` in entry-point scripts, not library modules.
- Use `logger = logging.getLogger(__name__)` in all modules.
