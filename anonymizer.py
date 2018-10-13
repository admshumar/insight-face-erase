# Given an image and a mask, use the mask to anonymize
# objects of interest in the image.

import os, re, glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from  Rectangle import Mask

#%matplotlib inline
from Rectangle import Mask # for testing

# Apply a NumPy array mask to a NumPy array image
def invert_mask(mask):
    inverter_array = 255 * np.ones(mask.shape)
    mask = inverter_array - mask
    return mask

def apply_mask(image, mask):
    return np.bitwise_and(image, mask)

# Apply an anonymization procedure to a NumPy array
def anonymize_image(image, mask, kernel_size):
    X_faces = apply_mask(image, mask)
    X_faces = cv2.GaussianBlur(X_faces, kernel_size)

    mask_complement = invert_mask(mask)
    mask_complement = cv2.GaussianBlur(X_faces, kernel_size)
    X_nonfaces = apply_mask(image, mask_complement)

    Y = X_faces + X_nonfaces

    return Y


def show_applied_mask(image, mask):
    X = apply_mask(image, mask)
    plt.imshow(X, cmap="gray")
    plt.show()

image = np.random.randn(256,256)
mask = Mask(256, 256, 64, 64, 192,  192)

