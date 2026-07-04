# PCA — Face Recognition (Principal Component Analysis)

This folder contains a small PCA-based face recognition.

## Repository layout

- `train.py` — Train a PCA model on the dataset and save the trained model.
- `test.py` — Run inference / evaluation using a trained PCA model.

## Prerequisites

- Python 3.8+ recommended.
- Install project dependencies from the repository root:

```bash
pip install -r requirements.txt
```

If you prefer a minimal set, ensure these packages are available: `numpy`, `scikit-learn`, `matplotlib`, `joblib`.

## Dataset

The example uses the Olivetti faces dataset included in the workspace at `web_app/assets/datasets/olivetti_faces_dataset.npz`.

## Quick Usage

From the repository root run (examples):

Train a PCA model:

```bash
# Using uv
uv run PCA\train.py     # For Windows
uv run PCA/train.py     # For Linux/MacOS

# Using pip
python PCA\train.py     # For Windows
python3 PCA/train.py    # For Linux/MacOS
```

Evaluate / test a trained model:

```bash
# Using uv
uv run PCA\test.py  # For Windows
uv run PCA/test.py  # For Linux/MacOS

# Using pip
python PCA\test.py  # For Windows
python3 PCA/test.py # For Linux/MacOS
```

Notes:
- `train.py` will load the dataset, compute PCA, train the classifier (if applicable), and save model artifacts (check the script for the exact output path). This uses Scikit Learn's PCA class, if needed a from scratch implementation is available in the `application/` directory of [Singular-Value-Decomposition](https://github.com/OjasRane/Singular-Value-Decomposition) repository. This from scratch implementation uses SVD.
- `test.py` will load saved model artifacts and produce evaluation metrics and example visualizations.

## Outputs

Model artifacts are typically saved under `PCA/`