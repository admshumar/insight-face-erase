import os, re, glob
import cv2
import numpy as np
from PascalVOCParse import PascalVOCAnnotation as Annotation
from Rectangle import Mask


image_main_path = "WIDER_images/*/images/*"
directories = glob.glob(image_main_path)

for directory in directories:
    masks = glob.glob(directory+"/*_mask.jpg")

    count = 0
    for mask in masks:
        os.remove(mask)
        count+=1

    print(str(count), "mask(s) removed.")
    del masks