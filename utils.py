# Classes for image and mask manipulation.

import os
import re
import glob
import cv2
import torch
import models
import numpy as np
from parser import Annotation


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


# Prepares image data for processing by UNet.
class Loader:

    path_string = "WIDER_images/*/images/*"
    image_directories = glob.glob(path_string)
    image_files = glob.glob(path_string+"/*.jpg")
    annotation_train_directory = "WIDER_images/WIDER_annotations/WIDER_train"
    annotation_val_directory = "WIDER_images/WIDER_annotations/WIDER_val"
    # annotation_test_directory = "WIDER_images/WIDER_annotations/WIDER_test"

    def __init__(self, dimension):
        self.dimension = dimension

    @classmethod
    def open_image_monochrome(cls, file):
        x = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        return x

    @classmethod
    def get_mask_directories(cls):
        for directory in Loader.image_directories:
            if not os.path.exists(directory+"/masks"):
                os.makedirs(directory+"/masks")
        return glob.glob(Loader.path_string + "/masks")

    @classmethod
    def match_masks_to_images(cls):
        mask_files = glob.glob(Loader.path_string + "/*_mask.jpg")

        print("Matching masks to images.")
        count = 0
        for mask in mask_files:
            image_file = re.sub(re.compile("_mask.jpg"), ".jpg", mask)
            if not os.path.exists(image_file):
                os.remove(mask)
                count = count + 1

        print(str(count) + " mask file(s) removed.")

    # Given a set of image paths, construct its corresponding set of mask paths
    @classmethod
    def get_mask_paths(cls, image_paths):
        mask_paths = []
        for image_path in image_paths:
            mask_path = image_path.split("/")
            mask_path = "/".join(mask_path[0:4]) + "/masks/" + mask_path[4]
            mask_paths.append(mask_path)
        return mask_paths

    # Given a set of image paths and a set of mask paths, pair each image with its mask and return the desired batch
    @classmethod
    def get_batch(cls, image_paths, batch_size, batch, seed):
        start = batch * batch_size
        end = min((batch + 1) * batch_size, len(image_paths))

        np.random.seed(seed)
        np.random.shuffle(image_paths)
        image_paths_batch = image_paths[start:end]

        masks_images = []
        mask_paths = Loader.get_mask_paths(image_paths_batch)
        for i in range(0, len(image_paths_batch)):
            x = Loader.open_image_monochrome(image_paths_batch[i])
            y = Loader.open_image_monochrome(mask_paths[i])
            masks_images.append([x, y])
        masks_images = np.asarray(masks_images)

        return masks_images

    # Construct image masks from parsed Pascal VOC annotations, then write as .jpg files
    def make_masks(self):
        directory = self.annotation_train_directory
        filepaths = glob.glob(directory + "/*.xml")

        print("Making masks.")

        for file in filepaths:  # for each .xml annotation file . . .
            # Get bounding box and path information from a list of .xml files in a directory.
            file_annotation = Annotation(file)
            file_name = re.sub(re.compile(".jpg"), "", file_annotation.path)
            file_name = re.sub(re.compile(r"\./"), "WIDER_images/", file_name)
            mask_name = file_name + "_mask.jpg"
            mask_name = mask_name.split("/")
            mask_name = "/".join(mask_name[0:4])+"/masks/"+mask_name[4]

            # Use Rectangle.Mask to construct masks from bounding box info.
            voc_mask = Mask(file_annotation.height, file_annotation.width, file_annotation.boxes)

            # Use cv2 to write masks as .jpg files to a separate image directory.
            cv2.imwrite(mask_name, voc_mask.inverted_array)  # CHANGE directory to images directory

    def resize_images(self):
        new_directory = "WIDER_images_"+str(self.dimension)+"/"
        main_directory = re.compile("WIDER_images/")
        directories = Loader.image_directories
        print("Resizing images.")

        # Create filepaths for newly resized images
        new_image_files = []
        for image_file in Loader.image_files:
            new_image_file = re.sub(main_directory, new_directory, image_file)
            new_image_files.append(new_image_file)

        # Replicate the WIDER_FACE directory structure for newly resized images
        for image_directory in directories:
            new_image_directory = re.sub(main_directory, new_directory, image_directory)
            if not os.path.exists(new_image_directory):
                os.makedirs(new_image_directory, exist_ok=True)

        # Write the resized images
        dim = (self.dimension, self.dimension)
        for image_directory in Loader.image_directories + glob.glob(Loader.path_string + "/masks"):
            files = glob.glob(image_directory+"/*.jpg")
            images = [cv2.imread(file) for file in files]

            for i in range(0, len(images)):
                new_path = re.sub(re.compile("WIDER_images/"), new_directory, files[i])
                im = cv2.resize(images[i], dim, interpolation=cv2.INTER_AREA)
                cv2.imwrite(new_path, im)
            del images

    def resize_masks(self):
        new_directory = "WIDER_images_"+str(self.dimension)+"/"
        main_directory = re.compile("WIDER_images/")
        directories = glob.glob(Loader.path_string + "/masks")
        print("Resizing images.")

        # Create filepaths for newly resized masks
        new_image_files = []
        for mask_file in glob.glob(Loader.path_string + "/masks/*.jpg"):
            new_mask_file = re.sub(main_directory, new_directory, mask_file)
            new_image_files.append(new_mask_file)

        # Replicate the WIDER_FACE directory structure for newly resized masks
        for image_directory in directories:
            new_image_directory = re.sub(main_directory, new_directory, image_directory)
            if not os.path.exists(new_image_directory):
                os.makedirs(new_image_directory, exist_ok=True)

        # Write the resized images
        dim = (self.dimension, self.dimension)
        for image_directory in Loader.image_directories + glob.glob(Loader.path_string + "/masks"):
            files = glob.glob(image_directory+"/*.jpg")
            images = [cv2.imread(file) for file in files]

            for i in range(0, len(images)):
                new_path = re.sub(re.compile("WIDER_images/"), new_directory, files[i])
                im = cv2.resize(images[i], dim, interpolation=cv2.INTER_AREA)
                cv2.imwrite(new_path, im)
            del images

    # In case there are more masks than images, delete the extraneous masks

    def rename_masks(self):
        new_directory = "WIDER_images_" + str(self.dimension) + "/"
        new_mask_paths = glob.glob(new_directory+"*/images/*/masks/*.jpg")
        for mask_file in new_mask_paths:
            os.rename(mask_file, re.sub(re.compile("_mask.jpg"), ".jpg", mask_file))

    def match_images_to_masks(self):
        new_directory = "WIDER_images_" + str(self.dimension) + "/"
        new_mask_paths = glob.glob(new_directory+"*/images/*/masks/*.jpg")

        print("Matching masks to images.")
        count = 0
        for mask in new_mask_paths:
            image_file = mask.split("/")
            image_file = "/".join(image_file[0:3])+"/"+image_file[5]
            if not os.path.exists(image_file):
                os.remove(mask)
                count += 1

        print(str(count) + " mask file(s) removed.")

    # Invert all masks
    def invert_masks(self):
        mask_path = "WIDER_images_" + str(self.dimension) + "/*/images/*/masks"
        filepaths = glob.glob(mask_path + "/*.jpg")

        for file in filepaths:
            im = cv2.imread(file)
            cv2.imwrite(file, cv2.bitwise_not(im))


# Post-processing of image masks outputs from UNet.
class Editor:

    @classmethod
    def apply_mask(cls, image, mask):
        return np.bitwise_and(image, mask)

    @classmethod
    def resize_mask(cls, image, height, width):
        image = cv2.resize(image, (height, width), interpolation=cv2.INTER_LINEAR)
        return image

    @classmethod
    def smooth_mask(cls, image, kernel_size=81):
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return image

    # Apply a NumPy array mask to a NumPy array image
    @classmethod
    def invert_mask(cls, mask):
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        inverter_array = np.max(mask) * np.ones(mask.shape)
        mask = inverter_array - mask
        return mask

    # Occlude objects in an image
    @classmethod
    def occlude_image(cls, image, mask):
        x = Editor.apply_mask(image, mask)
        return x

    # Apply an anonymization procedure to a NumPy array
    @classmethod
    def inpaint_occluded_image(cls, image, mask):
        im = cv2.inpaint(image, mask, 10, cv2.INPAINT_TELEA)
        return im

    # From the distribution of pixel intensities, select
    # the least pixel intensity in the highest bin, then
    # use that value as a threshold for the image.
    @classmethod
    def make_binary_mask(cls, image, scalar):
        row_length = image.shape[0]
        column_length = image.shape[1]

        distribution = np.histogram(image)
        threshold = distribution[1][len(distribution[1])-2]

        new_mask = np.zeros(image.shape)
        for i in range(0, row_length):
            for j in range(0, column_length):
                if image[i, j] > threshold:
                    new_mask[i, j] = scalar

        return new_mask

    @classmethod
    def make_binary_mask_from_torch(cls, image, scalar):
        image = image.detach().cpu().numpy()
        image = np.squeeze(image)
        height, width = image.shape

        new_mask = Editor.make_binary_mask(image, scalar)
        new_mask = new_mask.reshape(1, 1, height, width)
        new_mask = torch.from_numpy(new_mask)

        return new_mask.float()

    @classmethod
    def reshape_for_display(cls, i, list_of_images):
        x = list_of_images[i]
        if type(x) is torch.Tensor:
            x = x.detach().cpu().numpy()
        x = np.reshape(x, [256, 256])
        return x

    # Return the intersection over union of two NumPy arrays
    @classmethod
    def intersection_over_union(cls, y, z):
        iou = (np.sum(np.minimum(y, z))) / (np.sum(np.maximum(y, z)))
        return iou

    def __init__(self, image_paths, seed_index):
        self.image_paths = image_paths
        self.seed_index = seed_index
        self.samples = Loader.get_batch(self.image_paths, len(self.image_paths), 0, self.seed_index)
        self.samples_images = self.samples[:, 0]
        self.samples_masks = self.samples[:, 1]
        self.model = models.UNet()

    def initiate_model(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def get_raw_masks(self):
        with torch.no_grad:
            x = self.model(self.samples_images)
        return x

    def get_input(self, i, samples_images):
        return self.reshape_for_display(i, samples_images)

    def get_output(self, i, samples_images):
        y = samples_images[i]
        y = y.reshape(1, 1, y.shape[0], y.shape[1])
        y = torch.from_numpy(y)
        if torch.cuda.is_available():
            y = y.cuda()
        y = y.float()
        y = self.model(y)
        y = y.detach().cpu().numpy()
        y = np.reshape(y, [256, 256])
        return y
