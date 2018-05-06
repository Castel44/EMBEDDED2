import numpy as np
import hpelm
import time
from sklearn.preprocessing import OneHotEncoder
import ImageNet.load_ImageNet_data as Dataset


def classificator_HPELM(in_size, out_size, hyperpar):
    print('\nBuilding model: HPELM with hyperpar:\n ', hyperpar)
    model = hpelm.HPELM(in_size, out_size, classification="c", batch=hyperpar['batch_size'], accelerator="GPU",
                        precision='single', tprint=5, norm=hyperpar['norm'])
    model.add_neurons(hyperpar['n_neurons'], hyperpar['neuron_type'])
    print(str(model))
    t = time.time()
    model.train('hX_train.h5', 'hy_train.h5', 'c')
    elapsed_time_train = time.time() - t
    print("Training time: %f" % elapsed_time_train)
    model.predict('hX_train.h5', 'hy_train_pred.h5')
    print('Training Accuracy: ', (1 - model.error('hy_train.h5', 'hy_train_pred.h5')))
    model.predict('hX_test.h5', 'hy_test_pred.h5')
    print('Test Accuracy: ', (1 - model.error('hy_test.h5', 'hy_test_pred.h5')))


def loadImageNet(img_size):
    # Loading Dataset
    print('Loading ImageNet dataset')
    data_dir = "F:\\Documenti 2\\University\\Magistrale\\Progettazione Sistemi Embedded\\Progetto EMBEDDED\\Datasets\\"
    x_train, y_train, x_test, y_test = Dataset.load_ImageNet(data_dir, img_size=img_size)
    x_train /= x_train.std()
    print('X_train shape ', x_train.shape)
    print('y_train shape ', y_train.shape)
    print('X_test shape ', x_test.shape)
    print('y_test shape ', y_test.shape)
    out_class = len(np.unique(y_test))
    print('Num Classes: ', out_class)
    print('X_train mean: ', x_train.mean())
    print('X_train std: ', x_train.std())

    # Create hdf5 files of dataset
    print('Create HDF5 data')
    onehot = OneHotEncoder(sparse=False)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    # creating data on SSD, add data_dir to filepath to create on HDD
    hpelm.make_hdf5(x_train.reshape(-1, img_size*img_size*3), "hX_train.h5")
    hpelm.make_hdf5(onehot.fit_transform(y_train), "hy_train.h5")
    hpelm.make_hdf5(x_test.reshape(-1, img_size*img_size*3), "hX_test.h5")
    hpelm.make_hdf5(onehot.fit_transform(y_test), "hy_test.h5")

    del x_train, y_train, x_test, y_test


#######################################################################################################################
# HYPERPARAMETERS
image_size = 16
channels = 3
input_size = image_size*image_size*channels
out_class = 1000

hyperpar_dic = {
    'n_neurons': 15000,
    'neuron_type': 'sigm',
    'batch_size': 10000,
    'norm': 0.0001
}

# comment if dataset already loaded in SSD
#loadImageNet(image_size)

# suppose X and Y save on SSD with default names
# TODO: change dir for X and Y
classificator_HPELM(input_size, out_class, hyperpar_dic)
'''15000/0.001 -> 27,0.001'''