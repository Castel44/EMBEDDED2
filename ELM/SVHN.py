import numpy as np
import h5py
import os
import time
import hpelm
import matplotlib.pyplot as plt



def plot_images(images, nrows, ncols, cls_true, cls_pred=None):
    """ Plot nrows * ncols images from images and annotate the images
    """
    # Initialize the subplotgrid
    fig, axes = plt.subplots(nrows, ncols)

    # Randomly select nrows * ncols images
    rs = np.random.choice(images.shape[0], nrows * ncols)

    # For every axes object in the grid
    for i, ax in zip(rs, axes.flat):

        # Predictions are not passed
        if cls_pred is None:
            title = "True: {0}".format(np.argmax(cls_true[i]))

        # When predictions are passed, display labels + predictions
        else:
            title = "True: {0}, Pred: {1}".format(np.argmax(cls_true[i]), cls_pred[i])

            # Display the image
        ax.imshow(images[i, :, :, :])

        # Annotate the image
        ax.set_title(title)

        # Do not overlay a grid
        ax.set_xticks([])
        ax.set_yticks([])



#Loading preprocessed Data
path = os.getcwd() + '/data'

# Open the file as readonly
h5f = h5py.File(path+ '/SVHN_single.h5', 'r')

# Load the training, test and validation set
X_train = h5f['X_train'][:]
y_train = h5f['y_train'][:]
X_test = h5f['X_test'][:]
y_test = h5f['y_test'][:]
X_val = h5f['X_val'][:]
y_val = h5f['y_val'][:]

# Close this file
h5f.close()

print('Training set', X_train.shape, y_train.shape)
print('Validation set', X_val.shape, y_val.shape)
print('Test set', X_test.shape, y_test.shape)

# We know that SVHN images have 32 pixels in each dimension
img_size = X_train.shape[1]

# Greyscale images only have 1 color channel
num_channels = X_train.shape[-1]

# Number of classes, one class for each of 10 digits
num_classes = y_train.shape[1]

# Plot 2 rows with 9 images each from the training set
plot_images(X_train, 2, 9, y_train)


