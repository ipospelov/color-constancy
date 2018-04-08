import math
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import sRGBColor, LabColor

def euclidean_distance(x1, x2):
    distance = 0
    for i in range(len(x1)):
        distance += pow((x1[i] - x2[i]), 2)

    return math.sqrt(distance)

def ciede_distance(x1, x2):
    color1_lab = LabColor(x1[0], x1[1], x1[2])
    color2_lab = LabColor(x2[0], x2[1], x2[2])
    return delta_e_cie2000(color1_lab, color2_lab);