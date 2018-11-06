# Given an image and a mask, use the mask to anonymize
# objects of interest in the image.

import torch

# import os, re, glob
import glob
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

# Argument parser
import argparse

# UNet
from models import UNet
from utils import Loader
from utils import Editor

# Get arguments
parser = argparse.ArgumentParser(description='Test UNet on WIDER FACE Dataset')

parser.add_argument('--batchsize', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 10)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training (default: False)')
parser.add_argument('--datadir', type=str, default="WIDER_images_256/WIDER_val/images/*/*.jpg",
                    help='Dataset Directory (Typically the default is used!)')
parser.add_argument('--writedir', type=str, default="outputs",
                    help='Output Directory')
parser.add_argument('--statedict', type=str, default="weights.pth",
                    help='Name of state dictionary to be loaded')

args = parser.parse_args()


class Anonymizer:

    @classmethod
    def check_for_numpy(cls, tensor):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        return tensor

    @classmethod
    def get_number_of_batches(cls, image_paths, batch_size):
        batches = len(image_paths) / batch_size
        if not batches.is_integer():
            batches = math.floor(batches) + 1
        return int(batches)

    @classmethod
    def apply_mask(cls, image, mask):
        return np.minimum(image, mask)

    @classmethod
    def anonymize_image(cls, image, mask):
        image = Anonymizer.check_for_numpy(image)
        image = image.reshape(image.shape[-2:])
        image = np.float32(image)

        mask = Anonymizer.check_for_numpy(mask)
        mask = mask.reshape(mask.shape[-2:])
        mask = np.uint8(mask)

        im = Anonymizer.apply_mask(image, mask)
        print(im.shape, mask.shape)
        im = cv2.inpaint(im, mask, 10, cv2.INPAINT_TELEA)

        return im

    @classmethod
    def show_applied_mask(cls, image, mask):
        x = Anonymizer.apply_mask(image, mask)
        plt.imshow(x, cmap="gray")
        plt.show()

    def __init__(self, batch_size, image_paths, write_path, state_dict):
        self.batch_size = batch_size
        self.image_paths = glob.glob(image_paths)
        self.batches = Anonymizer.get_number_of_batches(self.image_paths, self.batch_size)
        self.write_path = write_path
        self.model = UNet()
        self.state_dict = state_dict

    def process_batch(self, batch):
        # Grab a batch, shuffled according to the provided seed. Note that
        # i-th image: samples[i][0], i-th mask: samples[i][1]
        samples = Loader.get_batch(self.image_paths, self.batch_size, batch, None)
        samples.astype(float)
        # Cast samples into torch.FloatTensor for interaction with U-Net
        samples = torch.from_numpy(samples)
        samples = samples.float()

        # Cast into a CUDA tensor, if GPUs are available
        if torch.cuda.is_available():
            samples = samples.cuda()

        # Isolate images and their masks
        samples_images = samples[:, 0]
        samples_masks = samples[:, 1]

        # Reshape for interaction with U-Net
        samples_images = samples_images.unsqueeze(1)
        source = samples_images
        samples_masks = samples_masks.unsqueeze(1)

        # Run inputs through the model
        output = self.model(samples_images)

        # Clamp the target for proper interaction with BCELoss
        target = torch.clamp(samples_masks, min=0, max=1)

        del samples

        return source, output, target

    def anonymize(self):
        count = 0
        for batch in range(self.batches):
            source, output, target = self.process_batch(batch)
            binary_mask = Editor.make_binary_mask_from_torch(output[0, :, :, :], 1.0)
            inverted_binary_mask = Editor.invert_mask(binary_mask)
            anonymized_image = Anonymizer.anonymize_image(source, inverted_binary_mask)

            cv2.imwrite(self.write_path + "/anon_" + str(count) + ".jpg", anonymized_image)
            count += 1

            del batch, target, output, binary_mask, anonymized_image

    def set_cuda(self):
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def set_weights(self):
        buffered_state_dict = torch.load("weights/" + self.state_dict)
        self.model.load_state_dict(buffered_state_dict)
        self.model.eval()


anonymizer = Anonymizer(args.batchsize, args.datadir, args.writedir, args.statedict)
anonymizer.set_cuda()
anonymizer.set_weights()
anonymizer.anonymize()
