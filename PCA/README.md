# PCA — Face Recognition

This folder contains the Principal Component Analysis workflow for face recognition in this project. It follows the classic eigenfaces approach: project face images into a lower-dimensional space and compare them using distance in that PCA subspace.

## Included files

- `train.py` — trains a PCA model on the Olivetti Faces dataset and saves the learned model.
- `test.py` — loads a trained model and evaluates it on a selected face sample.

## Core idea

PCA finds the directions of maximum variance in the dataset and represents each face as a compact feature vector in that reduced space. In face recognition, this is the eigenfaces formulation: each image is approximated by a weighted combination of principal components, and matching is done by comparing reconstructed or projected representations.

## Dataset

The workflow uses the Olivetti Faces dataset, which contains 40 subjects with 10 grayscale images each at 64 x 64 resolution.

## Quick usage

From the project root:

### Train the PCA model

```bash
# Using uv
uv run PCA\train.py     # For Windows
uv run PCA/train.py     # For Linux/MacOS

# Using pip
python PCA\train.py     # For Windows
python3 PCA/train.py    # For Linux/MacOS
```

### Test a trained model

```bash
# Using uv
uv run PCA\test.py  # For Windows
uv run PCA/test.py  # For Linux/MacOS

# Using pip
python PCA\test.py  # For Windows
python3 PCA/test.py # For Linux/MacOS
```

## Notes

- The implementation uses scikit-learn's `PCA` to build the eigenfaces representation.
- The project also includes a custom LDA/Fisherfaces comparison in `LDA/`, which is useful for studying how a discriminative method differs from PCA.
- Model artifacts and evaluation outputs are saved as part of the training and testing workflow in this directory or the project assets used by the app.