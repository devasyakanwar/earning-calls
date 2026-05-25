# Multimodal Earnings Call Intelligence System — Full Repo Review

After reviewing all **40+ files** across the entire repository, here is a comprehensive analysis of what can be improved, organized by severity and category.

---

## 🔴 Critical Issues (Must Fix)

### 1. Fake Backtesting — `backtesting.py` is Simulating Results, Not Testing
[backtesting.py](file:///c:/Users/devasya/Desktop/earning-calls/src/evaluation/backtesting.py#L56-L63)

The backtest **fabricates signals** by copying actual returns and intentionally flipping one to simulate 75% accuracy. This means the "75% accuracy" and "+10.3% alpha" reported in the README are **not real results** — they're circular.

```python
# Line 60-63: This is fabrication, not backtesting
signals = np.sign(actual_returns)
if len(signals) >= 4:
    signals[0] = -signals[0]  # "Intentionally flip one signal"
```

> [!CAUTION]
> The README's headline metrics (75% accuracy, +10.3% alpha) are based on fabricated backtest signals. This undermines the credibility of the entire project.

**Fix:** Run actual model inference on the test set to generate signals, then backtest those real predictions.

---

### 2. Duplicate Speaker Classification Logic (DRY Violation)
The same speaker classification heuristics are copy-pasted in **three** separate files:

| File | Location |
|:---|:---|
| [segmentation.py](file:///c:/Users/devasya/Desktop/earning-calls/src/preprocessing/segmentation.py#L37-L53) | `OPERATOR_KEYWORDS`, `EXECUTIVE_TITLES`, `ANALYST_KEYWORDS` |
| [download_transcripts.py](file:///c:/Users/devasya/Desktop/earning-calls/scripts/download_transcripts.py#L45-L59) | Exact same lists + `classify_speaker_role()` |
| [segmentation.py](file:///c:/Users/devasya/Desktop/earning-calls/src/preprocessing/segmentation.py#L73-L101) | `classify_speaker_role()` reimplemented |

**Fix:** Consolidate into a single `src/preprocessing/speaker_classification.py` module and import everywhere.

---

### 3. `logging.basicConfig()` Called in Every Module
**Every single source file** calls `logging.basicConfig()` at module level. This is an anti-pattern — `basicConfig()` should only be called once in the entry point (`main()`), not in library modules. When multiple modules are imported, only the first call takes effect; the rest are silently ignored.

**Affected files:** All 17 feature modules, all 5 modeling modules, all 3 evaluation modules, all preprocessing modules, the dashboard, and all scripts.

**Fix:** Remove `logging.basicConfig()` from all library modules. Keep only `logger = logging.getLogger(__name__)`. Configure logging once in the entry-point scripts.

---

## 🟠 Significant Improvements

### 4. No `__main__.py` or CLI Entry Point
There's no unified way to run the pipeline. Each module has its own `if __name__ == "__main__"` block with hardcoded paths. There's no orchestration script that chains the full pipeline.

**Fix:** Create a `src/__main__.py` or `scripts/run_pipeline.py` that orchestrates the full pipeline with a proper CLI (using `argparse` or `click`).

---

### 5. Hardcoded 40/60 Split for Q&A Detection
[qa_pressure.py:L89-L92](file:///c:/Users/devasya/Desktop/earning-calls/src/features/qa_pressure.py#L89-L92)

The Q&A pressure extractor assumes the first 40% of segments are "prepared remarks" and the rest are Q&A. This is a rough heuristic that ignores the actual `segment_type` labels already computed by `segmentation.py`.

```python
split_idx = int(n * 0.4)
prepared = call_df.head(split_idx)
qa = call_df.tail(n - split_idx)
```

**Fix:** Use the actual `segment_type` column for splitting (it's already computed in the segments table).

---

### 6. Missing `streamlit` and `plotly` from `pyproject.toml`
[requirements.txt](file:///c:/Users/devasya/Desktop/earning-calls/requirements.txt#L46-L47) lists `streamlit==1.42.0` and `plotly==6.0.0`, but [pyproject.toml](file:///c:/Users/devasya/Desktop/earning-calls/pyproject.toml#L7-L35) does **not** include them. Since `pyproject.toml` is the canonical dependency file for `uv sync`, the dashboard can't run after a fresh setup.

**Fix:** Add `streamlit` and `plotly` to `pyproject.toml` dependencies. Also add `requests` and `joblib` which are similarly missing.

---

### 7. Dependency Version Drift Between `requirements.txt` and `pyproject.toml`
Several packages have mismatched specifiers:

| Package | `requirements.txt` | `pyproject.toml` |
|:---|:---|:---|
| `python-dotenv` | no pin | `>=1.2.2` |
| `datasets` | `>=3.0.0` | `>=3.0.0` ✅ |
| `requests` | listed | **missing** |
| `joblib` | listed | **missing** |
| `streamlit` | `1.42.0` | **missing** |
| `plotly` | `6.0.0` | **missing** |

**Fix:** Keep one source of truth. Prefer `pyproject.toml` and generate `requirements.txt` from it if needed.

---

### 8. CI Workflow is Minimal and Likely Broken
[python-app.yml](file:///c:/Users/devasya/Desktop/earning-calls/.github/workflows/python-app.yml)

- Uses outdated `actions/checkout@v3` and `actions/setup-python@v3` (current is v4).
- Installs from `requirements.txt` via `pip` but the project uses `uv` — inconsistent.
- No caching of dependencies.
- No linting, type checking, or code formatting checks.
- Heavy ML dependencies (PyTorch, transformers) are installed every run without caching — this will be extremely slow.

**Fix:** Update to v4 actions, add dependency caching, consider a lightweight test profile without heavy ML dependencies.

---

### 9. `sentiment` Device Hardcoded to `cpu` in Config
[text_config.yaml:L8](file:///c:/Users/devasya/Desktop/earning-calls/configs/text_config.yaml#L8)

The config has `device: "cpu"` but also supports `"auto"`. Should default to `"auto"` so GPU/MPS is used when available.

---

### 10. Unused Imports and Dead Code
- [text_uncertainty.py:L41](file:///c:/Users/devasya/Desktop/earning-calls/src/features/text_uncertainty.py#L41): `self.complexity_config` is loaded but never used.
- [segmentation.py:L2](file:///c:/Users/devasya/Desktop/earning-calls/src/preprocessing/segmentation.py#L17): `import re` is imported but never used.
- [audio_wav2vec2.py:L17](file:///c:/Users/devasya/Desktop/earning-calls/src/features/audio_wav2vec2.py#L17): `import yaml` is imported but never used.
- [audio_wav2vec2.py:L38](file:///c:/Users/devasya/Desktop/earning-calls/src/features/audio_wav2vec2.py#L38): `config_path` parameter is accepted but never used.

---

## 🟡 Code Quality & Architecture

### 11. Polars/Pandas Mix in Dashboard
[app.py:L72](file:///c:/Users/devasya/Desktop/earning-calls/src/dashboard/app.py#L72)

Data is loaded with Polars then immediately converted to Pandas (`.to_pandas()`). Since Streamlit now supports Polars natively, this conversion is unnecessary overhead.

---

### 12. No Error Handling for Missing Data Files in Dashboard
[app.py:L63](file:///c:/Users/devasya/Desktop/earning-calls/src/dashboard/app.py#L63)

The dashboard will crash with an unhandled exception if `multimodal_dataset.parquet` doesn't exist. There should be a graceful fallback or setup instructions.

---

### 13. `fill_null(0.0)` Used Indiscriminately
Multiple files ([audio_assembler.py:L75](file:///c:/Users/devasya/Desktop/earning-calls/src/features/audio_assembler.py#L75), [text_assembler.py:L89](file:///c:/Users/devasya/Desktop/earning-calls/src/features/text_assembler.py#L89), [interaction_assembler.py:L110](file:///c:/Users/devasya/Desktop/earning-calls/src/features/interaction_assembler.py#L110)) fill **all** nulls with `0.0`. This silently masks data quality issues and can bias models. For example, a missing sentiment score of 0.0 is very different from actual neutral sentiment.

**Fix:** Use column-specific imputation strategies (median, mean, or explicit "missing" indicators).

---

### 14. SQL Injection Risk in DB Upsert
[db.py:L84-L88](file:///c:/Users/devasya/Desktop/earning-calls/src/preprocessing/db.py#L84-L88)

```python
conn.execute(
    f"DELETE FROM {table} WHERE {primary_key} IN "
    f"(SELECT {primary_key} FROM _staging)"
)
```

Table and column names are injected directly into SQL via f-strings. While this is internal code (not user-facing), it's a bad practice. DuckDB doesn't support parameterized identifiers, but you should at least validate that `table` and `primary_key` match expected values.

---

### 15. No Type Checking or Linting Configuration
There's no `mypy.ini`, `ruff.toml`, `pyproject.toml [tool.ruff]`, or similar configuration. For a project of this size with extensive type hints, running mypy/pyright would catch real bugs.

**Fix:** Add `[tool.ruff]` and `[tool.mypy]` sections to `pyproject.toml`.

---

### 16. Large File Committed: `test.tar.gz` (78MB) and `uv.lock` (313KB)
[data/test.tar.gz](file:///c:/Users/devasya/Desktop/earning-calls/data/test.tar.gz) is 78MB. This should be in `.gitignore` or managed with Git LFS. The `data/` directory gitignore only covers `data/raw/` and `data/processed/`, not `data/*.tar.gz`.

---

### 17. `implementation_plan.md.resolved` is a Stale Artifact (47KB)
This massive file at the project root seems to be a leftover from planning. It should either be cleaned up or moved to a docs folder.

---

## 🟢 Feature & Design Improvements

### 18. Feature Extraction is Not Parallelized
All feature extractors (prosody, openSMILE, wav2vec2, text sentiment, uncertainty, specificity) process segments **sequentially** in a for-loop with tqdm. For large datasets, this is very slow.

**Fix:** Use `multiprocessing.Pool`, `joblib.Parallel`, or Polars `.map_batches()` for CPU-bound extractors. For GPU-bound (FinBERT, wav2vec2), batching is already partially done but could be improved.

---

### 19. No Reproducibility Seeds in Key Places
- [dataset_split.py](file:///c:/Users/devasya/Desktop/earning-calls/src/features/dataset_split.py): Splits are deterministic (sorted by date), which is good.
- [train_fusion.py](file:///c:/Users/devasya/Desktop/earning-calls/src/modeling/train_fusion.py): No `torch.manual_seed()` or `np.random.seed()` is set before training.
- [train_lightgbm.py](file:///c:/Users/devasya/Desktop/earning-calls/src/modeling/train_lightgbm.py#L70): Has `random_state=42` ✅

**Fix:** Add global seed setting in `train_fusion.py`.

---

### 20. Windows Compatibility: `setup.sh` is Bash-Only
[setup.sh](file:///c:/Users/devasya/Desktop/earning-calls/setup.sh) is a Bash script with macOS/Linux-specific logic. You're on Windows. There should be a `setup.ps1` or a cross-platform Python setup script.

---

### 21. No Data Validation Layer
There's no schema validation between pipeline stages. If `text_sentiment.py` produces a column named `sentiment` instead of `sentiment_score`, downstream modules will silently fail or produce wrong results. The `_validate_columns()` in `db.py` only validates a minimal set of columns.

**Fix:** Add a Pydantic model or Polars schema validation at each pipeline boundary.

---

### 22. Notebook is a Stub
[vansh_finbert_sandbox.ipynb](file:///c:/Users/devasya/Desktop/earning-calls/notebooks/vansh_finbert_sandbox.ipynb) is only 2.8KB — it's essentially empty/stub. Either populate it with useful exploration or remove it.

---

### 23. `CONTRIBUTING.md` References Specific People
[CONTRIBUTING.md](file:///c:/Users/devasya/Desktop/earning-calls/CONTRIBUTING.md#L6-L9) names specific team members (Devasya, Vansh, Aadi). For an open-source project, this should be generalized. If it's private, it's fine.

---

### 24. No `.env.example` File
The project uses `python-dotenv` (in requirements) and has `.env` in `.gitignore`, but there's no `.env.example` template showing what environment variables are expected (e.g., HuggingFace tokens for pyannote-audio).

---

### 25. `speaker_override.yaml` is Empty
[speaker_override.yaml](file:///c:/Users/devasya/Desktop/earning-calls/configs/speaker_override.yaml) has `overrides: {}` and is never referenced by any source code. Either implement the override logic in `segmentation.py` or remove this config.

---

## 📊 Summary Table

| Category | Count | Priority |
|:---|:---:|:---:|
| 🔴 Critical (fake data, DRY, anti-patterns) | 3 | Immediate |
| 🟠 Significant (deps, CI, config) | 7 | High |
| 🟡 Code Quality | 7 | Medium |
| 🟢 Feature/Design | 8 | Low |
| **Total** | **25** | — |

## Recommended Action Order

1. **Fix the backtesting** — it's the most misleading issue
2. **Consolidate speaker classification** — eliminates 200+ lines of duplication
3. **Fix dependency sync** between `pyproject.toml` and `requirements.txt`
4. **Remove `logging.basicConfig()`** from all library modules
5. **Use actual `segment_type`** instead of 40/60 heuristic in Q&A pressure
6. **Add linting/type-checking** tooling
7. **Create a unified pipeline runner**
8. Everything else

