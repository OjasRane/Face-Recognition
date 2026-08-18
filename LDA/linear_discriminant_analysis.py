import numpy as np
from scipy.linalg import eigh
from sklearn.decomposition import PCA

class LDA:
    """
    Linear Discriminant Analysis (LDA)
    Uses scipy.linalg.eigh for eigen-decomposition
    """
    def __init__(self, n_components):
        self.n_components = n_components
        self.class_ = None
        self.n_class_ = None
        self.mean_ = None
        self.n_features_ = None
        self.n_samples_ = None
        self.Sw = None
        self.Sb = None
        self.W_ = None
        self.pca = None
        self.projected_class_means_ = None
    def fit(self, X, y):
        if X.ndim != 2:
            raise ValueError("Input X must be 2D array")
        self.n_samples_, self.n_features_ = X.shape
        self.class_, counts = np.unique(y, return_counts=True)
        self.n_class_ = len(self.class_)

        if self.n_features_ > self.n_samples_ - self.n_class_:
            self.pca = PCA(n_components=min(self.n_samples_ - self.n_class_, self.n_features_))
            X = self.pca.fit_transform(X)
            self.n_features_ = min(self.n_samples_ - self.n_class_, self.n_features_)

        self.mean_ = np.empty((self.n_class_+1, self.n_features_))
        self.mean_[-1, :] = np.mean(X, axis=0)
        self.Sw = np.zeros((self.n_features_, self.n_features_))
        self.Sb = np.zeros((self.n_features_, self.n_features_))

        for i, class_ in enumerate(self.class_):
            self.mean_[i, :] = np.mean(X[y == class_, :], axis=0)
            X_class = X[y == class_, :]
            X_class_centered = X_class - self.mean_[i, :]
            self.Sw += X_class_centered.T @ X_class_centered
            diff = self.mean_[i] - self.mean_[-1, :]
            self.Sb += len(X_class) * np.outer(diff, diff)

        eps = 1e-8 * np.trace(self.Sw) / self.n_features_
        eigenvalues, eigenvectors = eigh(self.Sb, self.Sw + eps*np.eye(self.n_features_))
        max_components = self.n_class_ - 1
        order = np.argsort(eigenvalues)[::-1]
        if self.n_components > max_components:
            raise ValueError(
                f"n_components={self.n_components} exceeds max possible"
                f"{max_components}=n_class-1"
            )
        top = order[:self.n_components]
        self.W_ = eigenvectors[:, top]
        self.projected_class_means_ = (self.mean_[:-1] - self.mean_[-1]) @ self.W_
    def transform(self, X):
        if X.ndim != 2:
            raise ValueError("Input X must be 2D array")
        if self.pca is not None:
            X = self.pca.transform(X)
        return (X - self.mean_[-1]) @ self.W_
    def fit_transform(self, X, y):
        if X.ndim != 2:
            raise ValueError("Input X must be 2D array")
        self.fit(X, y)
        return self.transform(X)
    def inverse_transform(self, X):
        X = X @ np.linalg.pinv(self.W_) + self.mean_[-1]
        if self.pca is not None:
            X = self.pca.inverse_transform(X)
        return X
    def predict(self, X):
        if X.ndim != 2:
            raise ValueError("Input X must be 2D array")
        transformed = self.transform(X)
        dists = np.linalg.norm(transformed[:, None, :] - self.projected_class_means_[None, :, :], axis=2)
        return self.class_[np.argmin(dists, axis=1)]