import cv2
import matplotlib.pyplot as plt

sigma1 = 0.15
sigma2 = 0.1
lambda1 = 1
lambda2 = 10
gamma = 0.3

def get_luminance_map(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    lab_channel, _, _ = cv2.split(lab_image)
    return get_filtered_map(lab_channel)
    #return lab_channel

def get_filtered_map(image):
    first_layer = cv2.getGaborKernel((3,3), sigma1, 0, lambda1, gamma, 0)
    second_layer = cv2.getGaborKernel((9,9), sigma2, 0, lambda2, gamma, 0)
    filtered = cv2.filter2D(image, -1, first_layer)
    filtered = cv2.filter2D(filtered, -1, second_layer)
    return filtered

def get_corrected_image(image, threshold):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    map = get_luminance_map(image)
    height, width, depth = image.shape
    for i in range(0, height):
        for j in range(0, width):
            if(map[i, j] < threshold):
                lab_image[i, j, 0] = lab_image[i, j, 0] + map[i, j]
            else:
                lab_image[i, j, 0] = lab_image[i, j, 0] - map[i, j]
    return cv2.cvtColor(lab_image, cv2.COLOR_Lab2BGR)


image = cv2.imread("pictures/dataset/1.jpg")
result = get_corrected_image(image, 50)
# plt.figure(2)
# plt.clf()
# ax = plt.axes([0, 0, 1, 1])
# plt.axis('off')
# plt.imshow(map_)
# plt.show()

cv2.imshow('img', result)
cv2.waitKey(0)