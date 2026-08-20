# Analysis

This folder contains the notebook-based analysis for the face-recognition project.
Rather than being centered on a single algorithm, this project is organized as a face-recognition laboratory where multiple classical methods can be studied on the same dataset and compared side by side.

PCA was the first method explored here, and LDA/Fisherfaces was added later as a second approach for comparison. The analysis directory is meant to grow with more algorithms in the future, so visitors can explore how different techniques behave on the same face-recognition task.

## Contents

- `face_recognition_using_PCA.ipynb`: A walkthrough of PCA, eigenfaces, dataset preparation, training, testing, and evaluation on the Olivetti Faces dataset.
- `face_recognition_using_LDA.ipynb`: A companion notebook covering Linear Discriminant Analysis/Fisherfaces and how it differs from PCA as a discriminative method for classification.

## What the notebooks cover

- PCA fundamentals and how they apply to image data.
- LDA/Fisherfaces fundamentals and the role of class separation.
- Loading and preparing the Olivetti Faces dataset.
- Flattening 64 x 64 grayscale faces into feature vectors.
- Training and testing PCA and LDA workflows.
- Projecting faces into learned feature spaces and comparing recognition performance.
- Visualizing sample faces, eigenfaces, class separation, and evaluation results.

## Requirements

The notebooks use the same Python dependencies as the rest of the project, including:

- NumPy
- Matplotlib
- scikit-learn
- SciPy
- JupyterLab or VS Code notebooks

Install them with either `uv sync` or `pip install -e .` from the project root.

## How to run

Open either notebook in JupyterLab or VS Code and run the cells from top to bottom.

If you prefer the terminal, start JupyterLab from the project root and open the notebook from there so relative imports and paths resolve correctly.

## Relationship to the rest of the project

These notebooks are the explanatory companion to the algorithm-specific scripts in `PCA/` and `LDA/`, as well as the interactive demos in `web_app/`.
They are the best place to inspect the behavior of each method in detail before making changes to the training pipeline or app logic.

## Notes

- This project is intentionally structured as a comparison and learning space, not as a single-method implementation.
- PCA provides the baseline variance-driven approach.
- LDA/Fisherfaces adds a class-aware alternative for comparison.
- More algorithms can be added here as the project grows.