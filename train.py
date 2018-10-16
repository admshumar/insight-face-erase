# File management
import glob

# Argument parser
import argparse

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

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
parser = argparse.ArgumentParser(description='UNet for WIDER FACE Dataset')

parser.add_argument('--batchsize', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 10)')
parser.add_argument('--train', action='store_true', default=False,
                    help='Argument to train model (default: False)')  # REMOVE
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--mom', type=float, default=0.99, metavar='MOM',
                    help='SGD momentum (default=0.99)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training (default: False)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='batches to wait before logging training status')  # REMOVE
parser.add_argument('--size', type=int, default=256, metavar='N',
                    help='imsize')
parser.add_argument('--seed', type=int, default=None, metavar='N',
                    help='NumPy Seed')
parser.add_argument('--datadir', type=str, default="WIDER_images_256/WIDER_train/images/*/*.jpg",
                    help='Dataset Directory (Typically the default is used!)')
parser.add_argument('--statedict', type=str, default="weights.pth",
                    help='Name of state dictionary for trained model')
parser.add_argument('--trainvalsplit', type=float, default=0.85,
                    help='Percent of input data to reserve for training')

args = parser.parse_args()


class Trainer:

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
        loss_2 = 1 - Trainer.intersection_over_union(output, target)
        loss = loss_1 + 0.1 * loss_2
        return loss

    def __init__(self,
                 side_length,
                 batch_size,
                 epochs,
                 learning_rate,
                 momentum_parameter,
                 seed,
                 image_paths,
                 state_dict,
                 train_val_split):
        self.side_length = side_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum_parameter = momentum_parameter
        self.seed = seed
        self.image_paths = glob.glob(image_paths)
        self.batches = Trainer.get_number_of_batches(self.image_paths, self.batch_size)
        self.model = UNet()
        self.loader = Loader(self.side_length)
        self.state_dict = state_dict
        self.train_val_split = train_val_split
        self.train_size = int(np.floor((self.train_val_split * self.batches)))

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

    def train_model(self):
        self.model.train()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        iteration = 0
        best_iteration = 0
        best_loss = 10 ** 10

        losses_train = []
        losses_val = []

        iou_train = []
        average_iou_train = []
        iou_val = []
        average_iou_val = []

        print("BEGIN TRAINING")
        print("TRAINING BATCHES:", self.train_size)
        print("VALIDATION BATCHES:", self.batches - self.train_size)
        print("BATCH SIZE:", self.batch_size)
        print("EPOCHS:", self.epochs)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~")

        for k in range(0, self.epochs):
            print("EPOCH:", k + 1)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~")

            # Train
            for batch in range(0, self.train_size):
                iteration = iteration + 1
                output, target = self.process_batch(batch)
                loss = Trainer.evaluate_loss(criterion, output, target)

                # Aggregate intersection over union scores for each element in the batch
                for i in range(0, output.shape[0]):
                    binary_mask = Editor.make_binary_mask_from_torch(output[i, :, :, :], 1.0)
                    iou = Trainer.intersection_over_union(binary_mask, target[i, :, :, :].cpu())
                    iou_train.append(iou.item())
                    print("IoU:", iou.item())

                print("Batch", batch, "of", self.train_size)

                # Clear data to prevent memory overload
                del target
                del output

                # Clear gradients, back-propagate, and update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Record the loss value
                loss_value = loss.item()
                if best_loss > loss_value:
                    best_loss = loss_value
                    best_iteration = iteration
                losses_train.append(loss_value)

                if batch == self.train_size - 1:
                    print("LOSS:", loss_value)
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~")

            average_iou = sum(iou_train)/len(iou_train)
            print("Average IoU:", average_iou)
            average_iou_train.append(average_iou)
            Visualizer.save_loss_plot(average_iou_train, "average_iou_train.png")

            # Validate
            for batch in range(self.train_size, self.batches):
                output, target = self.process_batch(batch)
                loss = Trainer.evaluate_loss(criterion, output, target)

                for i in range(0, output.shape[0]):
                    binary_mask = Editor.make_binary_mask_from_torch(output[i, :, :, :], 1.0)
                    iou = Trainer.intersection_over_union(binary_mask, target[i, :, :, :].cpu())
                    iou_val.append(iou.item())
                    print("IoU:", iou.item())

                loss_value = loss.item()
                losses_val.append(loss_value)
                print("VALIDATION LOSS:", loss_value)
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
                del output
                del target

            average_iou = sum(iou_val) / len(iou_val)
            print("Average IoU:", average_iou)
            average_iou_val.append(average_iou)
            Visualizer.save_loss_plot(average_iou_val, "average_iou_val.png")

        print("Least loss", best_loss, "at iteration", best_iteration)

        torch.save(self.model.state_dict(), "weights/"+self.state_dict)


trainer = Trainer(args.size,
                  args.batchsize,
                  args.epochs,
                  args.lr,
                  args.mom,
                  args.seed,
                  args.datadir,
                  args.statedict,
                  args.trainvalsplit)

trainer.set_cuda()
trainer.set_seed()

print(args)

trainer.train_model()
