# LDA / Fisherfaces

This folder contains the Linear Discriminant Analysis workflow used for face recognition in this project. The implementation follows the classic Fisherfaces idea: reduce dimensionality with PCA, then apply LDA to maximize class separation.

## Included files

- `train.py` — trains the PCA + LDA pipeline on the Olivetti Faces dataset and saves the model.
- `test.py` — evaluates a saved model against a selected face sample.
- `linear_discriminant_analysis.py` — a standalone LDA implementation built from the classical Fisherfaces formulation.

## Core idea

LDA is a supervised method that tries to project data so that samples from the same person are close together while samples from different people are far apart. In face recognition, this is often combined with PCA to avoid the small-sample-size issue and to improve class separation.

## Dataset

The workflow uses the Olivetti Faces dataset, which contains 40 subjects with 10 grayscale images each at 64 x 64 resolution.

## Quick usage

From the project root:

### Train the LDA model

```bash
# Using uv
uv run LDA\train.py     # For Windows
uv run LDA/train.py     # For Linux/MacOS

# Using pip
python LDA\train.py     # For Windows
python3 LDA/train.py    # For Linux/MacOS
```

### Test a trained model

```bash
# Using uv
uv run LDA\test.py  # For Windows
uv run LDA/test.py  # For Linux/MacOS

# Using pip
python LDA\test.py  # For Windows
python3 LDA/test.py # For Linux/MacOS
```

## Notes

- The project uses `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` in the training pipeline.
- The custom implementation in `linear_discriminant_analysis.py` is included as a reference to the mathematical formulation behind Fisherfaces.
- This directory complements the PCA experiments in `PCA/` and helps compare variance-based and class-separation-based recognition methods.
