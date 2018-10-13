# Methods for parsing PASCAL VOC annotation and obtaining bounding box information.

from typing import List, Any
from bs4 import BeautifulSoup


class Annotation:

    @classmethod
    def parse_annotation(cls, file):
        # Open an xml annotation, then parse it with BeautifulSoup
        annotation = open(file, 'r')
        voc = BeautifulSoup(annotation, "lxml-xml")
        return voc

    @classmethod
    def get_dimensions(cls, file):
        voc = Annotation.parse_annotation(file)
        return [int(voc.size.width.string), int(voc.size.height.string)]

    @classmethod
    def get_box_coordinates(cls, file):
        voc = Annotation.parse_annotation(file)

        # From the parsed annotation, return a list of bounding box coordinates
        boxes = []
        for box in voc.find_all('bndbox'):
            # The strings below are of type bs4.element.NavigableString,
            # thus we need an integer cast.
            boxes.append(int(box.xmin.string))
            boxes.append(int(box.ymin.string))
            boxes.append(int(box.xmax.string))
            boxes.append(int(box.ymax.string))

        return boxes

    @classmethod
    def get_annotation_path(cls, file):
        voc = Annotation.parse_annotation(file)
        return voc.path.string

    # Construct scale independent box coordinates.
    @classmethod
    def make_normalized_boxes(cls, width, height, boxes):
        normalized_boxes: List[Any] = []

        if len(boxes) % 2 == 0:
            n = int(len(boxes)/2)
            for i in range(0, n):
                normalized_width = boxes[2*i]/width
                normalized_height = boxes[2*i+1]/height
                normalized_boxes.append(normalized_width)
                normalized_boxes.append(normalized_height)

        return normalized_boxes

    def __init__(self, file):
        self.file = file
        self.width = Annotation.get_dimensions(self.file)[0]
        self.height = Annotation.get_dimensions(self.file)[1]
        self.boxes = Annotation.get_box_coordinates(self.file)
        self.path = Annotation.get_annotation_path(self.file)
        self.normalized_boxes = Annotation.make_normalized_boxes(self.width, self.height, self.boxes)
