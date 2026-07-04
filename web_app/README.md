# Face Recognition Lab

## Overview

This directory contains the Streamlit interface for the face-recognition demo. It provides a small interactive application for exploring the PCA eigenfaces workflow, viewing example face matches, and navigating back to the project home page.

## Features

- **Landing Page**: Main entry point with navigation into the PCA demo and links to the project resources.
- **PCA Demo**: Interactive page for selecting a subject, viewing the test face, and checking the nearest training match.
- **Cached Assets**: Precomputed dataset and PCA model artifacts for fast page loads.
- **Confusion Matrix View**: Displays the saved PCA confusion matrix used to inspect model behavior.

## Project Structure

```text
web_app/
├── .streamlit/
│   └── config.toml         # Streamlit configuration
├── assets/
│   ├── datasets/
│   │   └── olivetti_faces_dataset.npz
│   └── models/
│       └── pca/
│           ├── confusion_matrix.png
│           └── pca.pkl
├── landing_page.py         # Main landing page with navigation
└── pages/
	 └── pca.py              # PCA eigenfaces demo page
```

## Installation

### Prerequisites

- Python 3.12 or higher
- See [requirements.txt](../requirements.txt) in the project root

### Setup

Navigate to the project root directory:

```bash
cd Face-Recognition
```

#### Using uv:

```bash
uv sync
```

#### Using pip:

1. Create a virtual environment:

	```bash
	python3 -m venv .venv
	source .venv/bin/activate
	```

2. Install dependencies:

	```bash
	pip install -r requirements.txt
	```

## Running the Application

From the project root directory, start the Streamlit app:

### Using uv:

```bash
uv run streamlit run web_app\landing_page.py    # For Windows
uv run streamlit run web_app/landing_page.py     # For Linux/MacOS
```

### Using pip:

```bash
streamlit run web_app\landing_page.py            # For Windows
streamlit run web_app/landing_page.py            # For Linux/MacOS
```

The application will open in your default web browser.

## Pages Overview

### Landing Page (`landing_page.py`)
The main entry point featuring navigation sections for the PCA demo, notebook, GitHub repository, and social links.

### PCA Page (`pages/pca.py`)
Interactive PCA/eigenfaces page that lets you:

- Select a subject and test image
- View the chosen face alongside the nearest training match
- Inspect whether the prediction is correct
- Review the saved confusion matrix

## Dependencies

The web app depends on:

- **streamlit**: Web framework
- **numpy**: Numerical computations
- **matplotlib**: Face visualization and plot rendering
- **joblib**: Loading the saved PCA model artifact

The app also relies on project data and model artifacts stored in `web_app/assets/`.

## Troubleshooting

- **Page routing does not work**: Run Streamlit from the project root, not from inside `web_app/`.
- **Missing asset files**: Verify that `web_app/assets/datasets/olivetti_faces_dataset.npz`, `web_app/assets/models/pca/pca.pkl` and `web_app/assets/models/confusion_matrix.png` exist.
- **Image display issues**: Ensure the saved model and dataset files were not moved or renamed.

## Related Resources

- [Project README](../README.md) - Overall project documentation
- [Analysis Notebooks](../analysis/) - PCA walkthrough and mathematical explanation
- [PCA Scripts](../PCA/) - Training and test scripts for the eigenfaces workflow