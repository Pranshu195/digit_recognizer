import random
import cv2
import numpy as np


def rotation(img, degrees, interpolation=cv2.INTER_LINEAR, value=0):
    if isinstance(degrees, list):
        if len(degrees) == 2:
            degree = random.uniform(degrees[0], degrees[1])
        else:
            degree = random.choice(degrees)
    else:
        degree = degrees
    # print("transforms rotation: ", img.shape)
    h, w = img.shape[0:2]
    center = (w / 2, h / 2)
    map_matrix = cv2.getRotationMatrix2D(center, degree, 1.0)

    img = cv2.warpAffine(
        img,
        map_matrix, (w, h),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=value)

    return img


def flip(img):
    # print("transforms flip: ", img.shape)
    return np.fliplr(img)


def random_flip(img):
    if random.random() < 0.5:
        return flip(img)
    else:
        return img


def normalize(img, mean, std=None):
    img = img - np.array(mean)[np.newaxis, np.newaxis, ...]
    if std is not None:
        img = img / np.array(std)[np.newaxis, np.newaxis, ...]
    return img


def blur(img, kenrel_size=(5, 5), sigma=(1e-6, 0.6)):

    img = cv2.GaussianBlur(img, kenrel_size, random.uniform(*sigma))
    return img


def random_blur(img, kenrel_size=(5, 5), sigma=(1e-6, 0.6)):
    # print("transforms blur: ", img.shape)
    # print("transforms blur type: ", img.dtype)
    
    if random.random() < 0.5:
        return blur(img, kenrel_size, sigma)
    else:
        return img