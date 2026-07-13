# Analysis

This folder contains the notebook-based analysis for the face-recognition project.
The current notebook focuses on a PCA-based eigenfaces workflow built on the Olivetti Faces dataset.

## Contents

- `face_recognition_using_PCA.ipynb`: A step-by-step walkthrough of principal component analysis, eigenfaces, dataset preparation, training, testing, and evaluation.

## What the notebook covers

- PCA fundamentals and how they apply to image data.
- Loading and preparing the Olivetti Faces dataset.
- Flattening 64 x 64 grayscale faces into 4096-length feature vectors.
- Training a PCA model with 50 components.
- Projecting faces into the PCA space and comparing them with Euclidean distance.
- Visualizing representative faces, eigenfaces, and evaluation results.

## Requirements

The notebook uses the same Python dependencies as the rest of the project, including:

- NumPy
- Matplotlib
- scikit-learn
- JupyterLab or VS Code notebooks

Install them with either `uv sync` or `pip install -e .` from the project root.

## How to run

Open `analysis/face_recognition_using_PCA.ipynb` in JupyterLab or VS Code and run the cells from top to bottom.

If you prefer the terminal, start JupyterLab from the project root and open the notebook from there so relative imports and paths resolve correctly.

## Relationship to the rest of the project

This notebook is the exploratory and explanatory companion to the PCA scripts in `PCA/` and the Streamlit app in `web_app/`.
It is the best place to inspect the PCA workflow in detail before changing the training or demo code.

## Notes

- The current analysis focuses on PCA only; Fisherfaces/LDA remains a possible future extension.