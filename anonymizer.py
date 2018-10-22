# Given an image and a mask, use the mask to anonymize
# objects of interest in the image.

# import os, re, glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import Editor


class Anonymizer:

    @classmethod
    def apply_mask(cls, image, mask):
        return np.bitwise_and(image, mask)

    @classmethod
    def anonymize_image(cls, image, mask):
        im = Anonymizer.apply_mask(image, mask)
        im = cv2.inpaint(im, mask, 10, cv2.INPAINT_TELEA)
        return im

    @classmethod
    def show_applied_mask(cls, image, mask):
        X = Anonymizer.apply_mask(image, mask)
        plt.imshow(X, cmap="gray")
        plt.show()

