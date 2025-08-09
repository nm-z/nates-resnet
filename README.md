## nates-resnet

This Python-based training pipeline leverages PyTorch and Optuna to train a custom Residual Neural Network (ResNet) architecture specifically optimized for high-accuracy regression tasks—particularly predicting temperature from dense Vector Network Analyzer (VNA) sensor data.

### Key Features:

* **GPU-Exclusive Training**: Enforces GPU usage for high-speed, large-scale data handling.
* **Custom ResNet Architecture**:

  * Specialized residual blocks with LayerNorm, dropout, and learnable residual scaling for enhanced training stability.
  * Flexible depth (number of blocks) and width (hidden dimensions).
* **Optuna Hyperparameter Optimization**:

  * Automatic tuning of model parameters (hidden layers, dropout, learning rate).
  * Dynamically optimized preprocessing steps (`VarianceThreshold`, `StandardScaler`, `SelectKBest`).
* **Robust Data Preprocessing**:

  * Parses and aligns timestamped VNA data and temperature readings.
  * Handles missing and irregular data gracefully.
* **Automated GPU Batch-Size Scaling**:

  * Prevents GPU memory overflow by dynamically adjusting batch sizes.
* **Mixed-Precision Training (AMP)**:

  * Increases performance and reduces GPU memory usage.
* **Advanced Logging & Reproducibility**:

  * Unified logging to both terminal and log files.
  * SHA-256-based model and code snapshotting for reproducibility.
* **Comprehensive Artifact Management**:

  * Saves trained models, preprocessing components, metrics, hyperparameters, and dataset statistics for immediate inference deployment.

---

### Workflow:

1. **Data Loading and Preprocessing**

   * Dense feature extraction from raw VNA CSV files.
   * Temporal alignment with temperature measurements.

2. **Hyperparameter Optimization (Optuna)**

   * Iterative training with automatic hyperparameter tuning.
   * Early stopping based on validation R² to optimize efficiency.

3. **Final Training & Validation**

   * Retrains optimal model configuration on combined training and validation datasets.
   * Evaluates final model performance rigorously on a reserved holdout set.

4. **Deployment-Ready Model Saving**

   * Prepares and saves all necessary artifacts for easy inference and future usage.

---

### Targeted Performance:

* Designed explicitly to achieve high regression accuracy (R² > 0.98).

---

## Usage:

Run the main script to initiate the entire pipeline:

```bash
python3 gpu_optuna_trainer.py
```

Ensure GPU availability and proper dataset directories on lines 421-426 before execution.
Default is :
```python
VNA_DIR = 'VNA-D4'
TEMP_CSV = 'temp_readings-D4.csv'
FEATURE_COLUMNS = [
    'Phase(deg)',
    'Rs',
]
```




