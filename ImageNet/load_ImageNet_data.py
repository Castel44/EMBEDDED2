import numpy as np
import pickle


def unpickle(file):
    '''Load byte data from file'''
    with open(file, 'rb') as f:
        data = pickle.load(f)   #encoding = 'latin-1' ???
        return data


def load_ImageNet_batch(data_dir, idx, img_size=16):
    """Return train_data, train_labels, for #idx batch from ImageNet downsampled dataset.
    Reads data for train and validation.
    Already applies scaling, range data (0,1) with mean 0.

    Inputs:
        img_size[int]: 8,16,32,64 size of image
        dir_path[str]: path of dataset, already unzipped
        idx[int]: batch number
    """

    print('... Loading ImageNet%d, batch %d/10' %(img_size, idx))

    data_file = 'Imagenet{}_train\\'.format(img_size)

    d = unpickle(data_dir + data_file + 'train_data_batch_{}'.format(idx))
    x = d['data']
    y = d['labels']
    mean_image = d['mean']

    x = x / np.float32(255)
    mean_image = mean_image / np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i - 1 for i in y]
    data_size = x.shape[0]

    x -= mean_image

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2 * img_size2], x[:, 2 * img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)
    x = np.rollaxis(x, 1, 4)

    y = np.array(y)

    return x, y


def load_validation_data(data_dir, mean_image, img_size=16):
    """Load the test data for downsampled ImageNet dataset

    Inputs:
        img_size[int]: 8,16,32,64 size of image
        data_dir[str]: path of dataset, already unzipped
        idx[int]: batch number
        mean_image: can be extracted from any training data file
    """

    print('... Loading ImageNet%d test_data' % img_size)

    test_file = 'Imagenet{}_val\\'.format(img_size)

    d = unpickle(data_dir + test_file + 'val_data')
    x = d['data']
    y = d['labels']
    x = x / np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = np.array([i-1 for i in y])

    # Remove mean (computed from training data) from images
    x -= mean_image

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)
    x = np.rollaxis(x, 1, 4)

    y = np.array(y)

    return x, y


def load_ImageNet(data_dir, img_size=16):
    """ Load the ImageNet downsampled dataset from HDD.

    Inputs:
        data_dir[str]: path of dataset directory
        img_size[int]: 8,16,32,64 size of image

    Returns Numpy Array
    """

    # Load train data
    train_data = None
    train_labels = []

    for i in range(1,11):
        x, y = load_ImageNet_batch(data_dir, i, img_size= img_size)
        if i == 1:
            train_data = x
            train_labels = y
        else:
            train_data = np.vstack((train_data, x))
            train_labels = np.append(train_labels, y)

    # Load test data
    # Extracting mean
    d = unpickle(data_dir + 'Imagenet{}_train\\'.format(img_size) + 'train_data_batch_1')
    mean_image = d['mean']

    test_data, test_labels = load_validation_data(data_dir, mean_image, img_size=img_size)

    return train_data, train_labels, test_data, test_labels


def main():
    """ Testing purpose only"""

    data_dir = "F:\\Documenti 2\\University\\Magistrale\\Progettazione Sistemi Embedded\\Progetto EMBEDDED\\Datasets\\"

    x_train, y_train, x_test, y_test = load_ImageNet(data_dir, img_size= 16)

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)


#main()










