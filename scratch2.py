##########

# Grab a batch of images
samples = Loader.get_batch(image_paths, len(image_paths), 0, seed_index)

# Isolate images and their masks
samples_images = samples[:, 0]
samples_masks = samples[:, 1]


# Reshape images for visualization
def reshape_for_display(i, list_of_images):
    X = list_of_images[i]
    # X = X.detach().cpu().numpy()
    X = np.reshape(X, [256, 256])
    return X


def get_input(i, samples_images):
    return reshape_for_display(i, samples_images)


def get_output(i, samples_images):
    Y = samples_images[i]
    Y = Y.reshape(1, 1, Y.shape[0], Y.shape[1])
    Y = torch.from_numpy(Y)
    if CUDA:
        Y = Y.cuda()
    Y = Y.float()
    Y = model(Y)
    Y = Y.detach().cpu().numpy()
    Y = np.reshape(Y, [256, 256])
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
            if Y[i, j] > p:
                Z[i, j] = 255.0

    return Z


def process_output_mask(i, percentile, samples_images):
    mask = get_output(i, samples_images)
    return process_mask(mask, percentile)


# Return the intersection over union of two NumPy arrays
def intersection_over_union(Y, Z):
    iou = (np.sum(np.minimum(Y, Z))) / (np.sum(np.maximum(Y, Z)))
    return iou


# In particular, we want the IoU of a given mask and the
# mask processed by the network
def check_iou(i, percentile, samples_images, samples_masks):
    Y = process_output_mask(i, percentile, samples_images)
    Z = reshape_for_display(i, samples_masks)
    return intersection_over_union(Y, Z)


# Plots that compare the network output to the input's mask label
def view_model_map(i, percentile, samples_images, samples_masks):
    Z = reshape_for_display(i, samples_masks)
    Y = get_output(i, samples_images)
    W = process_output_mask(i, percentile, samples_images)
    V = process_mask(Y, 95)
    U = cv2.GaussianBlur(V, (81, 81), 0)

    images = plt.figure(figsize=(10, 5))
    images.add_subplot(1, 2, 1)
    show_input(i, samples_images)
    images.add_subplot(1, 2, 2)
    plt.imshow(Z, cmap="gray")
    plt.show(block=True)
    del images

    images = plt.figure(figsize=(10, 5))
    images.add_subplot(1, 2, 1)
    plt.imshow(Y, cmap="gray")
    images.add_subplot(1, 2, 2)
    plt.imshow(W, cmap="gray")
    plt.show(block=True)
    del images

    images = plt.figure(figsize=(10, 5))
    images.add_subplot(1, 2, 1)
    plt.imshow(V, cmap="gray")
    images.add_subplot(1, 2, 2)
    plt.imshow(U, cmap="gray")
    plt.show(block=True)
    del images


def show_results():
    for i in range(0, 10):
        percentile = 80

        view_model_map(i, percentile, samples_images, samples_masks)
        IOU = check_iou(i, percentile, samples_images, samples_masks)

        Y = samples_masks[i]  # .unsqueeze_(1).detach().cpu().numpy()
        Z = get_output(i, samples_images)

        # print("Values of input mask:", np.unique(Y))
        print("Values of output mask:"), print(np.unique(Z))
        print("Percentile of output mask:", np.percentile(Z, 50))
        print("Intersection over Union:", IOU)


########################################################

# Apply a NumPy array mask to a NumPy array image
def invert_mask(mask):
    inverter_array = 255 * np.ones(mask.shape)
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