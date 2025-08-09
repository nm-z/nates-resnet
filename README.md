## nates-resnet

Short, GPU-only training and inference pipeline for a custom ResNet regressor that predicts temperature from VNA traces. The project performs dataset assembly from raw CSVs, feature construction, Optuna-driven hyperparameter search, final training, and artifact packaging for reproducible inference.

### Methodology
- **Data ingestion**: Reads VNA CSV files from `VNA-D4B` and temperature readings from `temp_readings-*.csv`. Timestamps are parsed from VNA filenames and aligned to the nearest temperature within 15 minutes.
- **Feature construction**: For each file, required VNA columns are extracted and concatenated into a fixed-length dense vector using a per-file minimum length heuristic (median fallback). NaNs/±inf are zeroed.
- **Labels**: Temperature `temp_c` is taken from the aligned reading.
- **Split**: 70% train, 15% validation (for Optuna), 15% holdout for final reporting.
- **Model**: `CustomResNet` — input projection → N residual blocks with LayerNorm, linear, dropout, and learnable residual scaling (`alpha`) → linear head. Single-target regression.
- **Preprocessing (optional)**: VarianceThreshold, StandardScaler, and SelectKBest are toggled/parameterized by Optuna and, if used, are saved for exact reproduction.
- **Optimization**: Optuna TPE maximizes validation R². Mixed precision (AMP) is used; batch size is auto-scaled to fit GPU memory. Optional `torch.compile` when available.
- **Final training**: Trains with best hyperparameters on train+val; reports true holdout R². Early stopping with a patience window after a minimum epoch count.
- **Artifacts & versioning**: Saves to `inference-ready/` including weights, params, metrics, data stats, enabled preprocessors, holdout arrays, and a code snapshot (`model/model_def.py`) with a content-hash `version.json` for provenance.

### Quickstart
```bash
python gpu_optuna_trainer.py
```
- Requires a GPU (CUDA/ROCm). The code will hard-fail if no GPU is available.
- Outputs artifacts under `inference-ready/`.

### Inference
```bash
python inference.py
```
- Loads `inference-ready/` artifacts, reconstructs preprocessing in the original order, restores the snapshot model definition, and reports holdout metrics.

### Data expectations
- VNA directory: `VNA-D4B` with per-trace CSVs.
- Temperature CSV: `temp_readings-D4B.csv` or `temp_readings-D5.csv` depending on `DATASET_NAME` in `gpu_optuna_trainer.py`.
- Required VNA columns depend on dataset config. Note: current `D5` config aliases missing `Rs` to `Xs` when only `Xs` exists; `D4B` uses four channels directly. For strict integrity workflows, use columns that exist physically or adjust config to hard-fail on missing columns.

### Artifacts (inference-ready/)
- `resnet_model.pth` — trained weights
- `model_params.json` — selected hyperparameters (includes preprocessing toggles)
- `metrics.json` — final holdout R² with code version hash
- `data_stats.json` — basic stats and shapes for sanity checks
- `var_thresh.pkl` | `scaler.pkl` | `kbest_selector.pkl` — saved preprocessors if enabled
- `X_holdout_raw.npy`, `y_holdout.npy` — holdout arrays for end-to-end verification
- `model/model_def.py`, `version.json` — exact model definition and provenance

### Notes
- This repository enforces GPU-only execution for both training and inference.
- Provenance is preserved via code snapshotting; inference prefers the snapshot model to avoid shape or API drift.


