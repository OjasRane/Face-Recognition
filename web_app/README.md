# Face Recognition Lab

This directory contains the Streamlit interface for the face recognition project. It brings together the interactive demos for the PCA and LDA workflows so the project can be explored visually in a browser.

## Overview

The app includes a landing page for navigation and separate demo pages for the trained recognition models. It is designed to make the project easier to understand by showing the input image, the nearest match, and the associated evaluation output for each method.

## Included pages

- `landing_page.py` — main entry page with navigation to the available demos and project links
- `pages/pca.py` — PCA/eigenfaces demo
- `pages/lda.py` — LDA/Fisherfaces demo

## Features

- Interactive selection of training and test examples
- Comparison of PCA and LDA recognition behavior
- Visual inspection of nearest matches
- Review of saved evaluation artifacts such as confusion matrices
- Project links and navigation from a simple landing page

## Project structure

```text
web_app/
├── .streamlit/
│   └── config.toml
├── assets/
│   ├── datasets/
│   │   └── olivetti_faces_dataset.npz
│   └── models/
│       ├── lda/
│       │   ├── confusion_matrix.png
│       │   └── lda.pkl
│       └── pca/
│           ├── confusion_matrix.png
│           └── pca.pkl
├── landing_page.py
├── README.md
└── pages/
    ├── lda.py
    └── pca.py
```

## Installation

### Prerequisites

- Python 3.10+
- Project dependencies from [requirements.txt](../requirements.txt)

### Using uv

```bash
cd Face-Recognition
uv sync
```

### Using pip

```bash
cd Face-Recognition
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Running the app

From the project root:

### Using uv

```bash
# Windows
uv run streamlit run web_app\landing_page.py

# Linux/MacOS
uv run streamlit run web_app/landing_page.py
```

### Using pip

```bash
# Windows
streamlit run web_app\landing_page.py

# Linux/MacOS
streamlit run web_app/landing_page.py
```

The app will open in your browser and load the landing page.

## Page descriptions

### Landing page

The landing page acts as the entry point to the project and links to the different demos and project resources.

### PCA page

The PCA page demonstrates the eigenfaces workflow and allows the user to inspect the test image against the closest training face and the related evaluation output.

### LDA page

The LDA page demonstrates the Fisherfaces workflow and lets users compare how the class-discriminative approach behaves relative to PCA on the same dataset.

## Dependencies

The app depends on:

- `streamlit`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `joblib`

It also uses the saved model and dataset artifacts located in `web_app/assets/`.

## Related resources

- [Project README](../README.md)
- [Analysis notebooks](../analysis/)
- [PCA scripts](../PCA/)
- [LDA scripts](../LDA/)