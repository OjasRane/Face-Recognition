import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

confusion_matrix = np.load("./confusion_matrix.npy")
fig, ax = plt.subplots(figsize=(16, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
disp.plot(ax=ax, cmap=plt.cm.Blues)
plt.show()