# LOAD CIFAR10
# data -- 10000x3072 numpy array of uint8S. Each row of the array stores a 32x32 color image. each batch
#           [1024: Red][1024: Green][1024: Blue] ==> 3027
#
# labels -- 10000, the range 0 - 9 [airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck]

import pickle

import matplotlib.pyplot as plt
import numpy as np


def unpickle(file):
    '''Load byte data from file'''
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='latin-1')
        return data


def load_cifar10_data(data_dir):
    '''Return train_data, train_labels, test_data, test_labels
    The shape of data is 32 x 32 x3'''
    train_data = None
    train_labels = []

    for i in range(1, 6):
        data_dic = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            train_data = data_dic['data']
        else:
            train_data = np.vstack((train_data, data_dic['data']))
        train_labels += data_dic['labels']

    test_data_dic = unpickle(data_dir + "/test_batch")
    test_data = test_data_dic['data']
    test_labels = test_data_dic['labels']

    train_data = train_data.reshape((len(train_data), 3, 32, 32))
    train_data = np.rollaxis(train_data, 1, 4)
    train_labels = np.array(train_labels)

    test_data = test_data.reshape((len(test_data), 3, 32, 32))
    test_data = np.rollaxis(test_data, 1, 4)
    test_labels = np.array(test_labels)

    return train_data, train_labels, test_data, test_labels


def plot_example():
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 7
    for label_ind, cls in enumerate(classes):
        idxs = np.where(train_labels == label_ind)[0]
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + label_ind + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(train_data[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()


data_dir = "F:\\Documenti 2\\University\\Magistrale\\Progettazione Sistemi Embedded\\Progetto EMBEDDED\\Datasets\\cifar-10-batches-py"

train_data, train_labels, test_data, test_labels = load_cifar10_data(data_dir)


def main():
    '''Testing purpose only'''

    print(train_data.shape)
    print(train_labels.shape)

    print(test_data.shape)
    print(test_labels.shape)

    # In order to check where the data shows an image correctly
    plt.imshow(train_data[2])
    plt.show()

    plot_example()


if __name__ == '__main__':
    main()
