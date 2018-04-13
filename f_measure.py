import cv2
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

def get_measure_value(human_image: np, system_image : np):
    return f1_score(human_image.ravel(), system_image.ravel(), average='weighted');
#
# def get_1d_array(image):
#     image_1d = image.ravel()


def main():
    image_human = cv2.imread("pictures/dataset/1_human.jpg")
    # _, image_human = cv2.threshold(image_human, 45, 255, cv2.THRESH_BINARY)
    # plt.imshow(image_human)
    # plt.show()
    image_system = cv2.imread("pictures/dataset/1_result_clustered.jpg")
    image_human = np.array(image_human)
    image_system = np.array(image_system)

    print(image_system.shape)
    print(image_human.shape)

    print(f1_score(image_human.ravel(), image_system.ravel(), average='weighted'))


if __name__ == "__main__":
    main()