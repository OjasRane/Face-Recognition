# Face Recognition

This repository demonstrates face recognition with a small set of classical linear techniques. The current implementation centers on a PCA-based eigenfaces workflow on the Olivetti Faces dataset, achieving 88.8% accuracy while capturing 88.038% variance (50 components). The structure is left open for future Fisherfaces work based on LDA.

## Results
Landing Page of [Face Recognition Lab](https://face-recognition-lab.streamlit.app)
![Landing Page](readme_assets/landing_page_screenshot.png)

### PCA model

Screenshots from [Face Recognition Lab](https://face-recognition-lab.streamlit.app)
![PCA Playground](readme_assets/pca_screenshot.png)

Confusion matrix
![Confusion Matrix](./web_app/assets/models/pca/confusion_matrix.png)

## Project Overview

- `analysis/face_recognition_using_PCA.ipynb`: A notebook walkthrough covering PCA, eigenfaces, training, matching, and evaluation on the Olivetti Faces dataset.
- `PCA/train.py`: Trains the PCA model on the Olivetti training split and exports the trained model.
- `PCA/test.py`: Loads the exported model and finds the closest training face for a selected test image.
- `web_app/landing_page.py`: Streamlit landing page that routes to the current face-recognition demo and related links.
- `web_app/pages/pca.py`: Interactive PCA demo for choosing a subject, viewing the test image, and checking the nearest match.
- `web_app/assets/models/pca/pca.pkl`: Serialized PCA model and projected training vectors used by the web app.
- `web_app/assets/models/pca/confusion_matrix.png`: Pre-rendered confusion matrix shown in the web app.
- `web_app/assets/datasets/olivetti_faces_dataset.npz`: Cached train/test split used by the app.
- `utils.py`: Shared constants for image shape, dataset size, and PCA configuration.<br>
More detailed READMEs are provided in each directory.

The current flow uses the Olivetti Faces dataset, which contains 40 subjects with 10 grayscale images each at 64 x 64 pixels. The project is best read as a face-recognition playground rather than a single fixed model, with PCA serving as the current baseline and Fisherfaces/LDA reserved for a future extension.

## Repository Structure

Project directory structure, using only git-tracked files:

```text
.
├── .gitignore
├── .python-version
├── PCA/
│   ├── README.md
│   ├── test.py
│   └── train.py
├── readme_assets/
│   ├── landing_page_screenshot.png
│   └── pca_screenshot.png
├── README.md
├── analysis/
│   ├── README.md
│   └── face_recognition_using_PCA.ipynb
├── pyproject.toml
├── requirements.txt
├── utils.py
├── uv.lock
└── web_app/
    ├── README.md
    ├── .streamlit/
    │   └── config.toml
    ├── assets/
    │   ├── datasets/
    │   │   └── olivetti_faces_dataset.npz
    │   └── models/
    │       └── pca/
    │           ├── confusion_matrix.png
    │           └── pca.pkl
    ├── landing_page.py
    └── pages/
        └── pca.py
```

- `PCA/`: Training and test scripts for the current eigenfaces workflow.
- `analysis/`: Notebook-based documentation and experimentation.
- `web_app/`: Streamlit interface, cached dataset, and saved model artifacts.
- Root files: Project metadata, dependency definitions, shared constants, and repository config.

## Dependencies

- `numpy`: Array processing.
- `matplotlib`: Visualizing faces, eigenfaces, and evaluation plots.
- `scikit-learn`: Olivetti dataset loading, PCA, and metrics.
- `streamlit`: Interactive demo UI.
- `jupyterlab`: Notebook exploration.

## Usage

### Clone this directory

```bash
git clone https://github.com/OjasRane/Face-Recognition.git
cd Face-Recognition
```

## Running this project:

### Using uv (Recommended):

Install uv if needed by visiting [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)

Run the following command to create a virtual environment and install dependencies.

```bash
uv sync
```

### Using pip:

#### Creating and activating virtual environment

For Windows:
```ps1
python -m venv venv
venv\Scripts\Activate.ps1
```

For MacOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the notebook

Open `analysis/face_recognition_using_PCA.ipynb` in JupyterLab or VS Code and run the cells top to bottom.

### Train the PCA model

```bash
# Using uv
uv run PCA\train.py     # For Windows
uv run PCA/train.py     # For Linux/MacOS

# Using pip
python PCA\train.py     # For Windows
python3 PCA/train.py    # For Linux/MacOS
```

This exports `model.pkl` to the current working directory.

### Test a single face match

```bash
# Using uv
uv run PCA\test.py  # For Windows
uv run PCA/test.py  # For Linux/MacOS

# Using pip
python PCA\test.py  # For Windows
python3 PCA/test.py # For Linux/MacOS
```

Enter a test index between 0 and 79 when prompted.

### Launch the web app

Run the Streamlit app from the project root directory so the page routing and asset paths resolve correctly:

```bash
# Using uv
uv run streamlit run web_app\landing_page.py    # For Windows
uv run streamlit run web_app/landing_page.py    # For Linux/MacOS

# Using pip
streamlit run web_app\landing_page.py   # For Windows
streamlit run web_app/landing_page.py   # For Linux/MacOS
```

## Notes

- PCA is effective here because the dataset is aligned and controlled, but it still captures variance rather than identity directly.
- This project uses `sklearn.decomposition.PCA`, for a from scratch implementation refer the `principal_component_analysis` module from the repository [Singular-Value-Decomposition](https://github.com/OjasRane/Singular-Value-Decomposition).
- The notebook is the best place to see how the eigenfaces approach behaves across matching and failure cases.
- Future Fisherfaces/LDA work can fit naturally beside the current PCA flow without changing the overall project shape.
- The app relies on the precomputed assets in `web_app/assets/` so it can run without retraining first.

## Usage of AI

- All modules have code written entirely by me.
- AI was used for better understanding of the concepts.
- Draft for READMEs were generated by AI and were modified by me, for better explanation.

## Live Demo

[GitHub Pages](https://ojasrane.github.io/Face-Recognition)

[Face Recognition Lab](https://face-recognition-lab.streamlit.app)