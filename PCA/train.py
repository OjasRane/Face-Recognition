import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
import joblib
from utils import N_SUBJECTS, TRAIN_PER_SUBJECT, IMAGES_PER_SUBJECT, IMG_SIZE, TOTAL_IMAGES, N_COMPONENTS

def load_train_dataset():
    """
    Loads data
    :return: ndarray of shape (TRAIN_PER_SUBJECT*N_SUBJECTS, IMG_SIZE)
    """
    dataset = fetch_olivetti_faces()
    images = dataset.data
    faces = np.empty((N_SUBJECTS, TRAIN_PER_SUBJECT, IMG_SIZE))
    for i in range(0, TOTAL_IMAGES, IMAGES_PER_SUBJECT):
        faces[i // IMAGES_PER_SUBJECT, :, :] = images[i:(i + TRAIN_PER_SUBJECT), :]
    train_dataset = faces.reshape(-1, IMG_SIZE)
    return train_dataset

def train(train_dataset, n_components):
    """
    Trains PCA model
    :param train_dataset: ndarray of shape (TRAIN_PER_SUBJECT*N_SUBJECTS, IMG_SIZE)
    :param n_components: integer greater than equal to 1 and less than equal to IMG_SIZE
    :return: PCA model and projected features
    """
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(train_dataset)
    return pca, projected

if __name__ == "__main__":
    train_dataset = load_train_dataset()
    pca, projected = train(train_dataset, N_COMPONENTS)

    payload = {
        "model": pca,
        "projected": projected
    }
    joblib.dump(payload, "./model.pkl")