import numpy as np
import joblib
from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
from utils import N_SUBJECTS, TRAIN_PER_SUBJECT, TEST_PER_SUBJECT, IMAGES_PER_SUBJECT, IMG_SIZE, IMG_SHAPE, TOTAL_IMAGES

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

if __name__ == "__main__":
    test_dataset, original_dataset = load_test_dataset()
    model = joblib.load("model.pkl")
    test_idx = int(input(r"Enter the index for test image (0 ≤ index ≤ 79): "))
    if not (0 <= test_idx <= 79):
        raise ValueError(f"Index must be between 0 and 79")
    test_image = test_dataset[test_idx, :].reshape(1, -1)
    class_ = model.predict(test_image)[0]
    fig, axes = plt.subplots(1, 2)

    axes[0].imshow(test_image.reshape(IMG_SHAPE), cmap="gray", interpolation="bilinear")
    axes[0].set_title(f"Test Face: Subject {test_idx // 2}")

    rng = np.random.default_rng()
    axes[1].imshow(original_dataset[class_ * TRAIN_PER_SUBJECT + rng.integers(0, 8), :].reshape(IMG_SHAPE), cmap="gray", interpolation="bilinear")
    axes[1].set_title(f"Predicted class: Subject {class_}")

    for ax in axes.flatten():
        ax.axis("off")

    plt.tight_layout()
    plt.show()