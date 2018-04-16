import cv2
from colormath.color_objects import LabColor, LCHabColor, BaseRGBColor
from colormath.color_conversions import convert_color
import numpy as np
import matplotlib.pyplot as plt

sigma1 = 0.15
sigma2 = 0.45
lambda1 = 1
lambda2 = 1
gamma = 1

def correct_chroma(image, shift):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    height, width, _ = image.shape
    for i in range(0, height):
        for j in range(0, width):
            lab = LabColor(lab_image[i, j, 0], lab_image[i, j, 1], lab_image[i, j, 2])
            lch = convert_color(lab, LCHabColor)
            lch.lch_c += shift
            lab = convert_color(lch, LabColor)
            #print(lab)
            lab_image[i, j] = [lab.lab_l, lab.lab_a, lab.lab_b]
            #print(lab_image[i,j])

    image = cv2.cvtColor(lab_image, cv2.COLOR_Lab2BGR)
    return image

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
    lab_image = np.array(lab_image, dtype=np.int64)
    map = get_luminance_map(image)
    height, width, _ = image.shape
    for i in range(0, height):
        for j in range(0, width):
            # if(map[i, j] < 140):
            #     lab_image[i, j, 0] += abs(lab_image[i, j, 0] - threshold)
            # elif(map[i, j] > 220):
            #     lab_image[i, j, 0] -= abs(lab_image[i, j, 0] - threshold) * 0.3
            lab_image[i, j, 0] = pow(lab_image[i, j, 0]/255,0.25) * 255
            #lab_image[i, j, 0] -= 100

    lab_image = threshold_(lab_image)
    lab_image = np.array(lab_image, dtype=np.uint8)
    return  cv2.cvtColor(lab_image, cv2.COLOR_Lab2BGR)

def threshold_(img):
    height, width, _ = img.shape
    for i in range(0, height):
        for j in range(0, width):
            if(img[i, j, 0] < 0):
                img[i, j, 0] = 0
            if(img[i, j, 0] > 255):
                img[i, j, 0] = 255
    return img