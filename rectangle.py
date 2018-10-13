#Construct nparray image masks from PASCAL VOC annotations.
import numpy as np
#import cv2


class Rectangle():

    def __init__(self, height, width):
        self.height = height
        self.width = width

class Mask(Rectangle):

    def make_mask(self, height, width, boxes):

        X = (255.0)*np.ones((height, width), dtype=float)

        for k in range(0, int(len(boxes)/4)):
            for j in range(boxes[4*k+1], boxes[4*k+3]):
                for i in range(boxes[4*k], boxes[4*k+2]):
                    X[j, i] = 0
        return X

    def make_mask_label(self, height, width, boxes):

        X = np.ones((height, width), dtype=float)

        for k in range(0, int(len(boxes)/4)):
            for j in range(boxes[4*k+1], boxes[4*k+3]):
                for i in range(boxes[4*k], boxes[4*k+2]):
                    X[j, i] = 0
        return X

    def make_inverted_mask(self, height, width, boxes):

        X = np.zeros((height, width), dtype=float)

        if len(boxes)%4==0:
            for k in range(0, int(len(boxes)/4)):
                for j in range(boxes[4*k+1], boxes[4*k+3]):
                    for i in range(boxes[4*k], boxes[4*k+2]):
                        X[j, i] = 255.0

        return X

    def make_inverted_mask_label(self, height, width, boxes):

        X = np.zeros((height, width), dtype=float)

        if len(boxes)%4==0:
            for k in range(0, int(len(boxes)/4)):
                for j in range(boxes[4*k+1], boxes[4*k+3]):
                    for i in range(boxes[4*k], boxes[4*k+2]):
                        X[j, i] = 1

        return X

    def __init__(self, height, width, boxes):
        Rectangle.__init__(self, height, width)
        self.boxes = boxes
        self.array = Mask.make_mask(self, height, width, boxes)
        self.array_label = Mask.make_mask_label(self, height, width, boxes)
        self.inverted_array = Mask.make_inverted_mask(self, height, width, boxes)
        self.inverted_array_label = Mask.make_inverted_mask_label(self, height, width, boxes)
