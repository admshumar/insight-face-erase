# File management
import glob

# Argument parser
import argparse

# PyTorch
import torch
import torch.nn as nn

# UNet
from models import UNet

# Math
import math

# NumPy
import numpy as np

# Custom data loader
from utils import Loader
from utils import Editor
from utils import Visualizer


# Get arguments
parser = argparse.ArgumentParser(description='Test UNet on WIDER FACE Dataset')

parser.add_argument('--batchsize', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 10)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training (default: False)')
parser.add_argument('--size', type=int, default=256, metavar='N',
                    help='imsize')
parser.add_argument('--seed', type=int, default=None, metavar='N',
                    help='NumPy Seed')
parser.add_argument('--datadir', type=str, default="WIDER_images_256/WIDER_val/images/*/*.jpg",
                    help='Dataset Directory (Typically the default is used!)')
parser.add_argument('--statedict', type=str, default="weights.pth",
                    help='Name of state dictionary to be loaded')

args = parser.parse_args()


class Tester:

    @classmethod
    def intersection_over_union(cls, y, z):
        iou = (torch.sum(torch.min(y, z))) / (torch.sum(torch.max(y, z)))
        return iou

    @classmethod
    def get_number_of_batches(cls, image_paths, batch_size):
        batches = len(image_paths) / batch_size
        if not batches.is_integer():
            batches = math.floor(batches) + 1
        return int(batches)

    @classmethod
    def evaluate_loss(cls, criterion, output, target):
        loss_1 = criterion(output, target)
        loss_2 = 1 - Tester.intersection_over_union(output, target)
        loss = loss_1 + 0.1 * loss_2
        return loss

    def __init__(self,
                 side_length,
                 batch_size,
                 seed,
                 image_paths,
                 state_dict):
        self.side_length = side_length
        self.batch_size = batch_size
        self.seed = seed
        self.image_paths = glob.glob(image_paths)
        self.batches = Tester.get_number_of_batches(self.image_paths, self.batch_size)
        self.model = UNet()
        self.loader = Loader(self.side_length)
        self.state_dict = state_dict

    def set_cuda(self):
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def set_seed(self):
        if self.seed is not None:
            np.random.seed(self.seed)

    def process_batch(self, batch):
        # Grab a batch, shuffled according to the provided seed. Note that
        # i-th image: samples[i][0], i-th mask: samples[i][1]
        samples = Loader.get_batch(self.image_paths, self.batch_size, batch, self.seed)
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
        samples_masks = samples_masks.unsqueeze(1)

        # Run inputs through the model
        output = self.model(samples_images)

        # Clamp the target for proper interaction with BCELoss
        target = torch.clamp(samples_masks, min=0, max=1)

        del samples

        return output, target

    def test_model(self):
        buffered_state_dict = torch.load("weights/" + self.state_dict)
        self.model.load_state_dict(buffered_state_dict)
        self.model.eval()

        criterion = nn.BCELoss()
        iou_test = []
        batch_iou_test = []
        losses_test = []

        for batch in range(self.batches):
            output, target = self.process_batch(batch)
            loss = Tester.evaluate_loss(criterion, output, target)

            for i in range(0, output.shape[0]):
                batch_iou = 0
                binary_mask = Editor.make_binary_mask_from_torch(output[i, :, :, :], 1.0)
                iou = Tester.intersection_over_union(binary_mask, target[i, :, :, :].cpu())

                iou_test.append(iou.item())
                print("TEST IoU:", iou.item())

                batch_iou += iou.item()

            batch_iou_test.append(batch_iou / output.shape[0])

            loss_value = loss.item()
            losses_test.append(loss_value)
            print("TEST LOSS:", loss_value)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
            del output
            del target

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~")

        average_iou = sum(iou_test) / len(iou_test)
        print("Average IoU:", average_iou)

        average_batch_iou = sum(batch_iou_test) / len(batch_iou_test)
        print("Average Batch IoU:", average_batch_iou)

        Visualizer.save_loss_plot(average_iou, "average_iou.png")
        Visualizer.save_loss_plot(average_batch_iou, "average_batch_iou.png")


tester = Tester(args.size,
                args.batchsize,
                args.seed,
                args.datadir,
                args.statedict)

tester.set_cuda()
tester.set_seed()

print(args)

tester.test_model()
