# Construct NumPy image masks from PASCAL VOC annotations.
import numpy as np


class Mask:

    def __init__(self, height, width, boxes):
        # Image dimensions.
        self.height = height
        self.width = width

        # Bounding box information. Each bounding box
        # is given by four coordinates . . . two for the
        # upper-left pixel and two for the lower-right.
        self.boxes = boxes
        self.number_of_boxes = len(self.boxes)//4
        self.boxes_area = self.get_area()
        self.average_area = self.get_average_area()

        # Mask information. Labels are element-wise normalizations
        # for input into loss functions (e.g. nn.BCELoss in PyTorch).
        self.array = Mask.make_mask(self)
        self.array_label = Mask.make_mask(self, label=True)
        self.inverted_array = Mask.make_mask(self, inverted=True)
        self.inverted_array_label = Mask.make_mask(self, inverted=True, label=True)

    def get_area(self):
        area = 0
        for k in range(0, len(self.boxes) // 4):
            area += (self.boxes[4*k+3] - self.boxes[4*k+1]) * (self.boxes[4*k+2] - self.boxes[4*k])
        return area

    def get_average_area(self):
        if self.number_of_boxes > 0:
            return self.boxes_area / self.number_of_boxes
        else:
            return -1

    def make_mask(self, scale_factor=255.0, inverted=False, label=False):
        if inverted is True:
            mask = np.zeros((self.height, self.width), dtype=float)
        else:
            if label is True:
                scale_factor = 1
            mask = scale_factor * np.ones((self.height, self.width), dtype=float)

        for k in range(0, len(self.boxes)//4):
            for j in range(self.boxes[4*k+1], self.boxes[4*k+3]):
                for i in range(self.boxes[4*k], self.boxes[4*k+2]):
                    if inverted is True:
                        if label is True:
                            scale_factor = 1
                        mask[j, i] = scale_factor
                    else:
                        mask[j, i] = 0
        return mask
