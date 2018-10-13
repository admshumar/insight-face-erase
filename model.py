#########
#IMPORTS#
#########

# File management
import os, glob, re

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

# OpenCV
import cv2

# Math
import math

# NumPy
import numpy as np

# MatPlotLib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Custom data loader
from processor import Loader

######
#CUDA#
######
CUDA = torch.cuda.is_available()
print("CUDA is available:", CUDA)
if CUDA:
    print("Number of CUDA devices:", torch.cuda.device_count())
    torch.cuda.empty_cache()
    torch.cuda.init()
    print("Total allocated memory:", torch.cuda.max_memory_allocated(0))
    print("Cached memory:", torch.cuda.memory_cached(0))
    print("Total cached memory:", torch.cuda.max_memory_cached(0))


# #################
# #HYPERPARAMETERS#
# #################
# side_length = 256
# epoch = 50
# batch_size = 10
# learning_rate = 1e-3
# momentum_parameter = 0 #0.99

######################
#NETWORK ARCHITECTURE#
######################

# class UNet(nn.Module):
#
#     # U-Net's Downward Convolutional Block.
#     # i: number of input channels
#     # j: number of output channels
#     def step_down_a(i,j):
#         maps = nn.Sequential(
#                 nn.Conv2d(in_channels=i, out_channels=j, kernel_size=3, stride=1, bias=True, padding=1),
#                 nn.ReLU(),
#                 nn.BatchNorm2d(j),
#                 nn.Conv2d(in_channels=j, out_channels=j, kernel_size=3, stride=1, bias=True, padding=1),
#                 nn.ReLU(),
#                 nn.BatchNorm2d(j)
#         )
#         return maps
#
#     # U-Net's downsampling block.
#     # j: number of input channels for batch normalization.
#     def step_down_b(j):
#         maps = nn.Sequential(
#                 nn.MaxPool2d(2, stride=2, padding=0),
#                 nn.ReLU(),
#                 nn.BatchNorm2d(j)
#         )
#         return maps
#
#     # U-Net's "bottom_out layers".
#     # i: number of input channels.
#     def bottom_out(i):
#         maps = nn.Sequential(
#             nn.Conv2d(in_channels=i, out_channels=2*i, kernel_size=3, stride=1, bias=True, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(2*i),
#             nn.Conv2d(in_channels=2*i, out_channels=2*i, kernel_size=3, stride=1, bias=True, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(2*i),
#             nn.ConvTranspose2d(in_channels=2*i, out_channels=i, kernel_size=2, stride=2, bias=True, padding=0)
#         )
#         return maps
#
#     # U-Net's Upward Convolutional Block.
#     # i: number of input channels.
#     # j: number of output channels following the first convolution.
#     # k: number of output channels following transpose convolution.
#     def step_up(i,j,k):
#         maps = nn.Sequential(
#             nn.ReLU(),
#             nn.BatchNorm2d(i),
#             nn.Conv2d(in_channels=i, out_channels=j, kernel_size=3, stride=1, bias=True, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(j),
#             nn.Conv2d(in_channels=j, out_channels=j, kernel_size=3, stride=1, bias=True, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(j),
#             nn.ConvTranspose2d(in_channels=j, out_channels=k, kernel_size=2, stride=2, bias=True, padding=0)
#         )
#         return maps
#
#     #U-Net's Output Segmentation Block.
#     # i: number of input channels.
#     # j: number of output channels following the first convolution.
#     # k: number of mask channels.
#     def segment_output(i,j,k):
#         maps = nn.Sequential(
#             nn.Conv2d(in_channels=i, out_channels=j, kernel_size=3, stride=1, bias=True, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(j),
#             nn.Conv2d(in_channels=j, out_channels=j, kernel_size=3, stride=1, bias=True, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(j),
#             nn.Conv2d(in_channels=j, out_channels=k, kernel_size=1, stride=1, bias=True, padding=0)
#         )
#         return maps
#
#     def __init__(self):
#
#         super().__init__()
#
#         self.normalize = nn.BatchNorm2d(1)
#
#         self.encode1_a = UNet.step_down_a(1, 64)
#         self.encode1_b = UNet.step_down_b(64)
#         self.encode2_a = UNet.step_down_a(64, 128)
#         self.encode2_b = UNet.step_down_b(128)
#         self.encode3_a = UNet.step_down_a(128, 256)
#         self.encode3_b = UNet.step_down_b(256)
#         self.encode4_a = UNet.step_down_a(256, 512)
#         self.encode4_b = UNet.step_down_b(512)
#
#         self.bottom_out = UNet.bottom_out(512)
#
#         self.decode4 = UNet.step_up(1024, 512, 256)
#         self.decode3 = UNet.step_up(512, 256, 128)
#         self.decode2 = UNet.step_up(256, 128, 64)
#         self.segment = UNet.segment_output(128, 64, 1)
#
#         self.activate = nn.Softmax(dim=2)
#
#         # Produce weights with He initialization.
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
# #             elif isinstance(m, nn.BatchNorm2d):
# #                 nn.init.constant_(m.weight, 1)
# #                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         x = self.normalize(x)
#
#         # Encoding blocks are partitioned into two steps, so that
#         # the a convolved image can be stored for a skip connection.
#         x = self.encode1_a(x)
#         x1 = x
#         x = self.encode1_b(x)
#
#         x = self.encode2_a(x)
#         x2 = x
#         x = self.encode2_b(x)
#
#         x = self.encode3_a(x)
#         x3 = x
#         x = self.encode3_b(x)
#
#         x = self.encode4_a(x)
#         x4 = x
#         x = self.encode4_b(x)
#
#         x = self.bottom_out(x)
#
#         # Decoding blocks are preceded by a skip connection. The
#         # connection is a concatenation rather than a sum.
#         x = torch.cat((x4, x), 1)
#         x = self.decode4(x)
#
#         x = torch.cat((x3, x), 1)
#         x = self.decode3(x)
#
#         x = torch.cat((x, x), 1)
#         #x = torch.cat((x2, x), 1)
#         x = self.decode2(x)
#
#         x = torch.cat((x, x), 1)
#         #x = torch.cat((x1, x), 1)
#         x = self.segment(x)
#
#         # nn.Softmax() normalizes across only one dimension.
#         # Hence the following:
#         s1 = x.size(0), x.size(1), x.size(2)*x.size(3) # Shape parameters for return
#         s2 = x.shape # Shape parameters for Softmax activation
#
#         x = x.view(s1) # Reshape for Softmax activation
#         x = self.activate(x)
#         x = x.view(s2) # Reshape for return
#
#         return x


# In[4]:


#########
#SAMPLES#
#########

# #Random seed
# seed_index = None
# np.random.seed(seed_index)
#
# #Location of WIDER_FACE images
# image_paths = glob.glob("WIDER_images_256/WIDER_train/images/*/*.jpg")
#
# #Instantiate loader for WIDER_FACE images of size (side_length)**2
# loader = Loader(side_length)
#
# #Number of batches
# batches = len(image_paths)/batch_size
# if type(batches)==float:
#     batches = math.floor(batches)+1
    
######    
#LOSS#
######

# Return the intersection over union of two NumPy arrays
# def intersection_over_union(Y,Z):
#     iou = (torch.sum(torch.min(Y, Z)))/(torch.sum(torch.max(Y, Z)))
#     return iou


# In[5]:


#######
#TRAIN#
#######

# model = UNet()
# if CUDA:
#     model = model.cuda()
#
# criterion = nn.BCELoss()
# #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum_parameter)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate) #Hyperparameters given above.
#
# iteration = 0
# best_iteration = 0
# best_loss = 10**10
#
# losses = []
#
# print("BEGIN TRAINING")
# print("BATCHES:", batches)
# print("BATCH SIZE:", batch_size)
# print("EPOCHS:", epoch)
# print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
# for k in range (0, epoch):
#     print("EPOCH:", k+1)
#     print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
#     for batch in range(0, 20):
#         iteration = iteration + 1
#
#         # Grab a batch, shuffled according to the provided seed. Note that
#         # i-th image: samples[i][0], i-th mask: samples[i][1]
#         samples = loader.get_batch(image_paths, batch_size, batch, seed_index)
#
#         # Cast samples into torch.FloatTensor for interaction with U-Net
#         samples = torch.from_numpy(samples)
#         samples = samples.float()
#
#         # Cast into a CUDA tensor, if GPUs are available
#         if CUDA:
#             samples = samples.cuda()
#
#         # Isolate images and their masks
#         samples_images = samples[:,0]
#         samples_masks = samples[:,1]
#
#         # Reshape for interaction with U-Net
#         samples_images = samples_images.unsqueeze(1)
#         samples_masks = samples_masks.unsqueeze(1)
#
#         # Run inputs through the model
#         print(samples_images.shape)
#         output = model(samples_images)
#
#         # Clamp the target for proper interaction with BCELoss
#         target = torch.clamp(samples_masks, min=0, max=1)
#
#         # Evaluate the loss
#         loss = criterion(output, target) + (0.2)*(1-intersection_over_union(output, target))
#
#         # Clear data to prevent memory overload
#         del target
#         del output
#         del samples
#
#         # Clear gradients, backpropagate, and update weights
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         # Record the loss value
#         loss_value = loss.item()
#         if best_loss > loss_value:
#             best_loss = loss_value
#             best_iteration = iteration
#
#     # Plot the loss value
#     # if iteration%batch_size==0:
#     losses.append(loss_value)
#     plt.plot(losses)
# #     if epoch % 5 == 0:
# #         running_losses = losses[-5:]
# #         average_loss = sum(running_losses)/len(running_losses)
# #         plt.subplot(212)
# #         plt.plot(average_loss)
#
#     # print("ITERATION:", iteration)
#     print("LOSS:", loss_value)
#     print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
#     if epoch%5==0:
#         plt.ylabel("Cross Entropy Loss")
#         plt.show()
#
# print("Least loss", best_loss, "at iteration", best_iteration)


# In[6]:


################
#POSTPROCESSING#
################

# Grab a batch of images
samples = loader.get_batch(image_paths, len(image_paths), 0, seed_index)

# Isolate images and their masks
samples_images = samples[:,0]
samples_masks = samples[:,1]

# Reshape images for visualization
def reshape_for_display(i, list_of_images):
    X = list_of_images[i]
    # X = X.detach().cpu().numpy()
    X = np.reshape(X, [256,256])
    return X

def get_input(i, samples_images):
    return reshape_for_display(i, samples_images)

def get_output(i, samples_images):
    Y = samples_images[i]
    Y = Y.reshape(1,1,Y.shape[0], Y.shape[1])
    Y = torch.from_numpy(Y)
    if CUDA:
        Y = Y.cuda()
    Y = Y.float()
    Y = model(Y)
    Y = Y.detach().cpu().numpy()
    Y = np.reshape(Y, [256,256])
    return Y

def show_input(i, samples_images):
    X = get_input(i, samples_images)
    plt.imshow(X, cmap="gray")
    
def show_output(i, samples_images):
    Y = get_output(i, samples_images)
    plt.imshow(Y, cmap="gray")

# Use a percentile threshold to map a U-Net output
# to a binary mask
def process_mask(mask, percentile):
    Y = mask
    p = np.percentile(Y, percentile)
    row_length = Y.shape[0]
    column_length = Y.shape[1]
    
    Z = np.zeros(Y.shape)
    for i in range(0, row_length):
        for j in range(0, column_length):
            if Y[i,j] > p:
                Z[i,j] = 255.0

    return Z

def process_output_mask(i, percentile, samples_images):
    mask = get_output(i, samples_images)
    return process_mask(mask, percentile)
    
# Return the intersection over union of two NumPy arrays
def intersection_over_union(Y,Z):
    iou = (np.sum(np.minimum(Y, Z)))/(np.sum(np.maximum(Y, Z)))
    return iou

# In particular, we want the IoU of a given mask and the
# mask processed by the network
def check_iou(i, percentile, samples_images, samples_masks):
    Y = process_output_mask(i, percentile, samples_images)
    Z = reshape_for_display(i, samples_masks)
    return intersection_over_union(Y,Z)

# Plots that compare the network output to the input's mask label
def view_model_map(i, percentile, samples_images, samples_masks):
    
    Z = reshape_for_display(i, samples_masks)
    Y = get_output(i, samples_images)
    W = process_output_mask(i, percentile, samples_images)
    V = process_mask(Y, 95)
    U = cv2.GaussianBlur(V, (81,81), 0)
    
    
    images = plt.figure(figsize=(10,5))
    images.add_subplot(1,2,1)
    show_input(i, samples_images)
    images.add_subplot(1,2,2)
    plt.imshow(Z, cmap="gray")
    plt.show(block=True)
    del images
    
    images = plt.figure(figsize=(10,5))
    images.add_subplot(1,2,1)
    plt.imshow(Y, cmap="gray")
    images.add_subplot(1,2,2)
    plt.imshow(W, cmap="gray")
    plt.show(block=True)
    del images
    
    images = plt.figure(figsize=(10,5))
    images.add_subplot(1,2,1)
    plt.imshow(V, cmap="gray")
    images.add_subplot(1,2,2)
    plt.imshow(U, cmap="gray")
    plt.show(block=True)
    del images

def show_results():
    for i in range(0,10):
        percentile = 80
        
        view_model_map(i, percentile, samples_images, samples_masks)
        IOU = check_iou(i, percentile, samples_images, samples_masks)

        Y = samples_masks[i]#.unsqueeze_(1).detach().cpu().numpy()
        Z = get_output(i, samples_images)

        # print("Values of input mask:", np.unique(Y))
        print("Values of output mask:"), print(np.unique(Z))
        print("Percentile of output mask:", np.percentile(Z, 50))
        print("Intersection over Union:", IOU)

########################################################        

# Apply a NumPy array mask to a NumPy array image
def invert_mask(mask):
    inverter_array = 255*np.ones(mask.shape)
    mask = inverter_array - mask
    return mask

def apply_mask(image, mask):
    return np.bitwise_and(image, mask)

# Apply an anonymization procedure to a NumPy array
def anonymize_image(image, mask, kernel_size):
    X_faces = apply_mask(image, mask)
    X_faces = cv2.GaussianBlur(X_faces, kernel_size)
    
    mask_complement = invert_mask(mask)
    mask_complement = cv2.GaussianBlur(X_faces, kernel_size)
    X_nonfaces = apply_mask(image, mask_complement)
    
    Y = X_faces + X_nonfaces
    
    return Y

# print(apply_mask())
show_results()


# In[6]:


###############
#SERIALIZATION#
###############
SERIALIZATION_DIRECTORY = "/parameters/"
#torch.save(model.state_dict(), "u_net_1.pth")


# In[ ]:


# def average_iou(samples_images, samples_masks):
#     c=0
#     for i in range(0, batch_size):
#         c += check_iou(i, samples_images, samples_masks)
#     c = c/batch_size
#     return c
print("Hello")


# In[11]:


############
#SAVE MODEL#
############
print("Hello")


# In[ ]:




