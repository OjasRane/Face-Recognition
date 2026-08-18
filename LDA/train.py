import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_olivetti_faces
import joblib
from utils import N_SUBJECTS, TRAIN_PER_SUBJECT, IMAGES_PER_SUBJECT, IMG_SIZE, TOTAL_IMAGES

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
    y = np.repeat(np.arange(N_SUBJECTS), TRAIN_PER_SUBJECT)
    return train_dataset, y

def train(train_dataset, labels, lda__n_components=20, pca__n_components=150):
    """
    Trains pipeline of PCA and LDA. The default arguments for n_components gives a 98.75% accuracy.
    :param train_dataset: ndarray of shape (TRAIN_PER_SUBJECT*N_SUBJECTS, IMG_SIZE)
    :param labels: ndarray of shape (TRAIN_PER_SUBJECT*N_SUBJECTS)
    :param lda__n_components: n_components for LinearDiscriminantAnalysis
    :param pca__n_components: n_components for PCA
    :return: Trained model
    """
    pca = PCA(n_components=pca__n_components)
    lda = LinearDiscriminantAnalysis(n_components=lda__n_components)
    model = Pipeline([('pca', pca), ('lda', lda)])
    model.fit(train_dataset, labels)
    return model

if __name__ == "__main__":
    X, y = load_train_dataset()
    model = train(X, y)
    joblib.dump(model, "model.pkl")