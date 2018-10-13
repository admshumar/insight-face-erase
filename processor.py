# A loader class that includes image and mask manipulation.

import os
import re
import glob
import cv2
import numpy as np
from parser import Annotation
from rectangle import Mask


class Loader:

    path_string = "WIDER_images/*/images/*"
    image_directories = glob.glob(path_string)
    image_files = glob.glob(path_string+"/*.jpg")
    annotation_train_directory = "WIDER_images/WIDER_annotations/WIDER_train"
    annotation_val_directory = "WIDER_images/WIDER_annotations/WIDER_val"

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
