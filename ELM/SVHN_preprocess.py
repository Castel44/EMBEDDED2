import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import hpelm

datapath = "F:\\Documenti 2\\University\\Magistrale\\Progettazione Sistemi Embedded\\Progetto EMBEDDED\\Datasets\\SVHN"

def load_data(path):
    """ Helper function for loading a MAT-File"""
    data = loadmat(path)
    return data['X'], data['y']


def plot_images(img, labels, nrows, ncols):
    """ Plot nrows x ncols images
    """
    fig, axes = plt.subplots(nrows, ncols)
    for i, ax in enumerate(axes.flat):
        if img[i].shape == (32, 32, 3):
            ax.imshow(img[i])
        else:
            ax.imshow(img[i,:,:,0])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(labels[i])


def balanced_subsample(y, s):
    """Return a balanced subsample of the population"""
    sample = []
    # For every label in the dataset
    for label in np.unique(y):
        # Get the index of all images with a specific label
        images = np.where(y==label)[0]
        # Draw a random sample from the images
        random_sample = np.random.choice(images, size=s, replace=False)
        # Add the random sample to our subsample list
        sample += random_sample.tolist()
    return sample


print('Loading SVHN data')
X_train, y_train = load_data(datapath + '/train_32x32.mat')
X_test, y_test = load_data(datapath + '/test_32x32.mat')
X_extra, y_extra = load_data(datapath + '/extra_32x32.mat')
print("Training", X_train.shape, y_train.shape)
print("Test", X_test.shape, y_test.shape)
print('Extra', X_extra.shape, y_extra.shape)

# Transpose columns [image, row, column, colorchannel]
X_train, y_train = X_train.transpose((3,0,1,2)), y_train[:,0]
X_test, y_test = X_test.transpose((3,0,1,2)), y_test[:,0]
X_extra, y_extra = X_extra.transpose((3,0,1,2)), y_extra[:,0]
print("Training", X_train.shape)
print("Test", X_test.shape)
print("Extra", X_extra.shape)
print('')

# Calculate the total number of images
num_images = X_train.shape[0] + X_test.shape[0] + X_extra.shape[0]
print("Total Number of Images", num_images)


# Plot some training set images
plot_images(X_train, y_train, 2, 8)
# Plot some test set images
plot_images(X_test, y_test, 2, 8)
# Plot some extra set images
plot_images(X_extra, y_extra, 2, 8)

#update labels
y_train[y_train == 10] = 0
y_test[y_test == 10] = 0
y_extra[y_extra == 10] = 0


# Pick 400 samples per class from the training samples
train_samples = balanced_subsample(y_train, 400)
# Pick 200 samples per class from the extra dataset
extra_samples = balanced_subsample(y_extra, 200)

X_val, y_val = np.copy(X_train[train_samples]), np.copy(y_train[train_samples])

# Remove the samples to avoid duplicates
X_train = np.delete(X_train, train_samples, axis=0)
y_train = np.delete(y_train, train_samples, axis=0)

X_val = np.concatenate([X_val, np.copy(X_extra[extra_samples])])
y_val = np.concatenate([y_val, np.copy(y_extra[extra_samples])])

# Remove the samples to avoid duplicates
X_extra = np.delete(X_extra, extra_samples, axis=0)
y_extra = np.delete(y_extra, extra_samples, axis=0)

X_train = np.concatenate([X_train, X_extra])
y_train = np.concatenate([y_train, y_extra])
X_test, y_test = X_test, y_test

print("Training", X_train.shape, y_train.shape)
print("Test", X_test.shape, y_test.shape)
print('Validation', X_val.shape, y_val.shape)

# Sanity check
# Assert that we did not remove or add any duplicates
assert(num_images == X_train.shape[0] + X_test.shape[0] + X_val.shape[0])
# Display some samples images from the training set
plot_images(X_train, y_train, 2, 10)
# Display some samples images from the test set
plot_images(X_test, y_test, 2, 10)
# Display some samples images from the validation set
plot_images(X_val, y_val, 2, 10)


# One Hot Encoding
from sklearn.preprocessing import OneHotEncoder
# Fit the OneHotEncoder
enc = OneHotEncoder().fit(y_train.reshape(-1, 1))

# Transform the label values to a one-hot-encoding scheme
y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
y_test = enc.transform(y_test.reshape(-1, 1)).toarray()
y_val = enc.transform(y_val.reshape(-1, 1)).toarray()

print("Training set", y_train.shape)
print("Test set", y_test.shape)
print("Training set", y_val.shape)

'''
# Store in h5file
import h5py
savedir = os.getcwd() + '/data/'

try:
    os.mkdir(savedir)
except FileExistsError:
    print('DataDir already exist')


# Create file
h5f = h5py.File(savedir + 'SVHN_single.h5', 'w')

# Store the datasets
h5f.create_dataset('X_train', data=X_train)
h5f.create_dataset('y_train', data=y_train)
h5f.create_dataset('X_test', data=X_test)
h5f.create_dataset('y_test', data=y_test)
h5f.create_dataset('X_val', data=X_val)
h5f.create_dataset('y_val', data=y_val)

# Close the file
h5f.close()
'''

# Convert data in HDF5 files
hpelm.make_hdf5(X_train.reshape(-1, 32*32*3), "data/hX_train.h5")
hpelm.make_hdf5(y_train, "data/hy_train.h5")
hpelm.make_hdf5(X_test.reshape(-1, 32*32*3), "data/hX_test.h5")
hpelm.make_hdf5(y_test, "data/hy_test.h5")
hpelm.make_hdf5(X_val.reshape(-1, 32*32*3), "data/hX_val.h5")
hpelm.make_hdf5(y_val, "data/hy_val.h5")

hpelm.normalize_hdf5("data/hX_train.h5")
hpelm.normalize_hdf5("data/hX_test.h5")
hpelm.normalize_hdf5("data/hX_val.h5")