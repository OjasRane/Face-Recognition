import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

N_SUBJECTS = 40
TRAIN_PER_SUBJECT = 8
TEST_PER_SUBJECT = 2
IMAGES_PER_SUBJECT = 10
IMG_SIZE = 4096
IMG_SHAPE = (64, 64)
TOTAL_IMAGES = 400

def load_test_dataset():
    """
    Loads data
    :return: test_dataset: ndarray of shape (TEST_PER_SUBJECT*N_SUBJECTS, IMG_SIZE) and original_dataset: ndarray of shape (TOTAL_IMAGES, IMG_SIZE)
    """
    dataset = fetch_olivetti_faces()
    images = dataset.data
    test_faces = np.empty((N_SUBJECTS, TEST_PER_SUBJECT, IMG_SIZE))
    original_faces = np.empty((N_SUBJECTS, TRAIN_PER_SUBJECT, IMG_SIZE))
    for i in range(0, TOTAL_IMAGES, IMAGES_PER_SUBJECT):
        test_faces[i // IMAGES_PER_SUBJECT, :, :] = images[(i + TRAIN_PER_SUBJECT):(i + IMAGES_PER_SUBJECT)]
        original_faces[i // IMAGES_PER_SUBJECT, :, :] = images[i:(i + TRAIN_PER_SUBJECT)]
    test_dataset = test_faces.reshape(-1, IMG_SIZE)
    original_dataset = original_faces.reshape(-1, IMG_SIZE)
    return test_dataset, original_dataset

def matched_image(pca_model, test_image, original_dataset, projected):
    """
    Finds the image closest to the test face
    :param pca_model: sklearn.decomposition.PCA object
    :param test_image: ndarray of shape (1, IMG_SIZE)
    :param original_dataset: ndarray of shape (TRAIN_PER_SUBJECT*N_SUBJECTS, IMG_SIZE) consisting of all images from train dataset
    :param projected: ndarray of projected features with shape (TRAIN_PER_SUBJECT*N_SUBJECTS, N_COMPONENTS)
    :return: Image closest to the test face
    """
    projection_of_test_image = pca_model.transform(test_image)
    distance = np.linalg.norm(projected - projection_of_test_image, axis=1)
    min_distance_idx = np.argmin(distance)
    return original_dataset[min_distance_idx, :].reshape(IMG_SHAPE)

if __name__ == "__main__":
    test_dataset, original_dataset = load_test_dataset()
    payload = joblib.load("model.pkl")
    test_image_idx = int(input(r"Enter the index for test image (0 ≤ index ≤ 79): "))
    if not (0 <= test_image_idx <= TEST_PER_SUBJECT * N_SUBJECTS - 1):
        raise ValueError(f"Index must be between 0 and {TEST_PER_SUBJECT * N_SUBJECTS - 1}")
    test_image = test_dataset[test_image_idx].reshape(1, -1)
    closest_image = matched_image(
        payload["model"], test_image, original_dataset, payload["projected"]
    )

    figure, axes = plt.subplots(1, 2)

    axes[0].imshow(test_image.reshape(IMG_SHAPE), cmap="gray", interpolation="bilinear")
    axes[0].set_title("Test Face")

    axes[1].imshow(closest_image, cmap="gray", interpolation="bilinear")
    axes[1].set_title("Test face matched with")

    for ax in axes.flatten():
        ax.axis("off")

    plt.tight_layout()
    plt.show()