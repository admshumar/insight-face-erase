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
from statistics import mean

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
    def partition_masks(cls, output, target):
        # Partition the union of the output and target into a true positive mask,
        # a false positive mask, and a false negative mask
        true_positive_mask = torch.min(output, target)
        false_positive_mask = output - true_positive_mask
        false_negative_mask = target - true_positive_mask
        return true_positive_mask, false_positive_mask, false_negative_mask

    @classmethod
    def get_partition_measures(cls, output, target):
        true_positive_mask, false_positive_mask, false_negative_mask = Tester.partition_masks(output, target)

        tp = torch.sum(true_positive_mask) / (torch.sum(true_positive_mask) + torch.sum(false_positive_mask))
        fp = torch.sum(false_positive_mask) / (torch.sum(true_positive_mask) + torch.sum(false_positive_mask))
        fn = torch.sum(false_negative_mask) / (torch.sum(true_positive_mask) + torch.sum(false_negative_mask))

        return tp, fp, fn

    @classmethod
    def get_dice(cls, output, target):
        tp, fp, fn = Tester.get_partition_measures(output, target)
        if tp + fp + fn == 0:
            return -1
        dice = (2*tp)/(2*tp + fp + fn)
        if math.isnan(dice):
            return 0
        return dice.item()

    @classmethod
    def get_intersection_over_union(cls, output, target):
        tp, fp, fn = Tester.get_partition_measures(output, target)
        if tp + fp + fn == 0:
            return -1
        iou = tp / (tp + fp + fn)
        if math.isnan(iou):
            return 0
        return iou.item()

    @classmethod
    def get_accuracy(cls, output, target):
        tp, fp, fn = Tester.get_partition_measures(output, target)
        if tp + fp == 0:
            return -1
        accuracy = tp / (tp + fp)
        if math.isnan(accuracy):
            return 0
        return accuracy.item()

    @classmethod
    def get_recall(cls, output, target):
        tp, fp, fn = Tester.get_partition_measures(output, target)
        if tp + fn == 0:
            return -1
        recall = tp / (tp + fn)
        if math.isnan(recall):
            return 0
        return recall.item()

    @classmethod
    def get_number_of_batches(cls, image_paths, batch_size):
        batches = len(image_paths) / batch_size
        if not batches.is_integer():
            batches = math.floor(batches) + 1
        return int(batches)

    @classmethod
    def evaluate_loss(cls, criterion, output, target):
        loss_1 = criterion(output, target)
        loss_2 = 1 - Tester.get_intersection_over_union(output, target)
        loss = loss_1 + 0.1 * loss_2
        return loss

    def __init__(self, side_length, batch_size, seed, image_paths, state_dict):
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
        accuracy_count = 0
        image_count = 0
        accuracy_list = []
        recall_list = []
        iou_list = []
        dice_list = []
        batch_iou_list = []
        losses_list = []

        for batch in range(self.batches):
            output, target = self.process_batch(batch)
            loss = Tester.evaluate_loss(criterion, output, target)

            print("Batch:", batch)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
            batch_iou = 0

            for i in range(0, output.shape[0]):
                image_count += 1
                binary_mask = Editor.make_binary_mask_from_torch(output[i, :, :, :], 1.0)

                # Metrics
                accuracy = Tester.get_accuracy(binary_mask, target[i, :, :, :].cpu())
                recall = Tester.get_recall(binary_mask, target[i, :, :, :].cpu())
                iou = Tester.get_intersection_over_union(binary_mask, target[i, :, :, :].cpu())
                dice = Tester.get_dice(binary_mask, target[i, :, :, :].cpu())

                if accuracy == 1:
                    accuracy_count += 1

                accuracy_list.append(accuracy)
                recall_list.append(recall)
                iou_list.append(iou)
                dice_list.append(dice)

                print("Accuracy:", accuracy)
                print("Recall:", recall)
                print("IoU:", iou)
                print("Dice:", dice,"\n")
                print("Mean Accuracy:", mean(accuracy_list))
                print("Mean Recall:", mean(recall_list))
                print("Mean IoU:", mean(iou_list))
                print("Mean Dice:", mean(dice_list))
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~")

                batch_iou += iou

            print(batch_iou)
            print("Appending", batch_iou / output.shape[0])
            batch_iou_list.append(batch_iou / output.shape[0])
            print(batch_iou_list)

            loss_value = loss.item()
            losses_list.append(loss_value)
            print("Test loss:", loss_value)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
            del output
            del target

        mean_iou = mean(iou_list)
        mean_accuracy = mean(accuracy_list)
        mean_recall = mean(recall_list)
        mean_dice = mean(dice_list)

        print("Perfect Accuracy Percentage:", accuracy_count / image_count)
        print("Mean Accuracy:", mean_accuracy)
        print("Mean Recall:", mean_recall)
        print("Mean IoU:", mean_iou)
        print("Mean Dice:", mean_dice)


        average_batch_iou = sum(batch_iou_list) / len(batch_iou_list)
        print("Average Batch IoU:", average_batch_iou)

        Visualizer.save_loss_plot(iou_list, "iou_list.png")

        #Visualizer.save_loss_plot(mean_iou, "mean_iou.png")
        #Visualizer.save_loss_plot(average_batch_iou, "average_batch_iou.png")


tester = Tester(args.size,
                args.batchsize,
                args.seed,
                args.datadir,
                args.statedict)

tester.set_cuda()
tester.set_seed()

print(args)

tester.test_model()
