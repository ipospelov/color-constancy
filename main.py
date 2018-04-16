from preprocessing import get_corrected_image
import cv2
import numpy as np
from sklearn.utils import shuffle
import k_medoids
from utils.data_operations import recreate_image
import matplotlib.pyplot as plt

n_colors = 3
n_samples = 100
name = "3"
path = "../../dataset/"
extension = ".jpg"

def get_clustered_image(image, n_colors, n_train):
    preprocessed = get_corrected_image(image, 50)
    print("preprocessed")
    preprocessed = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2Lab)
    preprocessed = np.array(preprocessed, dtype=np.float32) / 100
    w, h, d = original_shape = tuple(preprocessed.shape)
    assert d == 3
    image_array = np.reshape(preprocessed, (w * h, d))
    image_array_sample = shuffle(image_array, random_state=0)[:n_train]
    clf = k_medoids.PAM(k=n_colors)
    clf.fit(image_array_sample)
    labels = clf.predict(image_array)
    final_image = recreate_image(clf.cluster_medoids_, labels, w, h)
    final_image = np.array(final_image, dtype=np.float32) * 100
    final_image = np.array(final_image, dtype=np.uint8)
    return cv2.cvtColor(final_image, cv2.COLOR_Lab2RGB)


image = cv2.imread(path + name + extension)
clustered = get_clustered_image(image, n_colors, n_samples)
plt.imshow(clustered)
plt.show()
final_image = cv2.cvtColor(clustered, cv2.COLOR_RGB2BGR)
cv2.imwrite(path + name + "_clustered" + extension, final_image)