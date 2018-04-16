import cv2
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

name = "lake"
path = "../../dataset/"
extension = ".jpg"
clustered = "_clustered"
boundaries = "_boundaries"
human = "_human"

def get_measure_value(human_image: np, system_image : np):
    return f1_score(human_image.ravel(), system_image.ravel(), average='weighted');

def main():

    image_human = cv2.imread(path + name + human + extension)
    image_system_clustered = cv2.imread(path + name + clustered + boundaries + extension)
    image_system = cv2.imread(path + name + boundaries + extension)
    image_human = np.array(image_human)
    image_system = np.array(image_system)
    image_system_clustered = np.array(image_system_clustered)

    print("without clustering:")
    print(f1_score(image_human.ravel(), image_system.ravel(), average='weighted'))
    print("\nwith clustering:")
    print(f1_score(image_human.ravel(), image_system_clustered.ravel(), average='weighted'))


if __name__ == "__main__":
    main()