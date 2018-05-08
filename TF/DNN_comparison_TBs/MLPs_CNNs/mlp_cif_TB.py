
from keras.callbacks import TensorBoard
import keras

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import InputLayer, Dense, Dropout, Activation, Flatten, BatchNormalization
from lr_sched import CyclicLR

import tensorflow as tf
import numpy as np
import keras
import os
import time
import itertools


def load_mnist():
    from keras.datasets import mnist
    print('Loading MNIST dataset')
    train, test = mnist.load_data()
    x_train, y_train = train
    x_test, y_test = test
    del train, test
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)
    x_train = x_train.reshape(-1,28,28,1).astype('float32')
    x_test = x_test.reshape(-1,28,28,1).astype('float32')
    img_size = 28
    img_channels = 1
    return x_train, x_test, y_train, y_test, img_size, img_channels


def load_cifar():
    from keras.datasets import cifar10
    print("Loading Dataset: CIFAR10")
    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)
    x_train = x_train.reshape(-1, 32 ,32, 3).astype('float32')
    x_test = x_test.reshape(-1, 32 , 32 , 3).astype('float32')
    img_size = 32
    img_channels = 3
    return x_train, x_test, y_train, y_test, img_size, img_channels


def load_SVHN():
    """RGB 32x32 SVHN dataset, stored on HDD"""
    import h5py
    path = "/home/sam/Downloads/svhn-multi-digit-master/data"
    print('Loading SVHN dataset...')
    # Open the file as readonly
    h5f = h5py.File(path + '/SVHN_single.h5', 'r')
    # Load the training, test and validation set
    x_train = h5f['X_train'][:]
    y_train = h5f['y_train'][:]
    x_test = h5f['X_test'][:]
    y_test = h5f['y_test'][:]
    h5f.close()
    img_size = 32
    img_channels = 3

    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)
    # Make train and test set a multiple of 100
    deleted_train = np.random.choice(y_train.shape[0], 388, replace=False)
    x_train = np.delete(x_train, deleted_train, axis=0)
    y_train = np.delete(y_train, deleted_train, axis=0)
    deleted_test = np.random.choice(y_test.shape[0], 32, replace=False)
    x_test = np.delete(x_test, deleted_test, axis=0)
    y_test = np.delete(y_test, deleted_test, axis=0)

    print('Training set', x_train.shape, y_train.shape)
    print('Test set', x_test.shape, y_test.shape)

    return x_train.astype('float32'), x_test.astype('float32'), y_train, y_test, img_size, img_channels








######################################################################################################################
# Get dataset
x_train, x_test, y_train, y_test, img_size, img_channels = load_SVHN()

# Data scaler
#from sklearn.preprocessing import StandardScaler
#prescaler = StandardScaler()
#x_train = prescaler.fit_transform(x_train)
#x_test = prescaler.transform(x_test)

######################################################################################################################
# Hyperparameters
input_size = [32,32,3]
output_size = 10
n_neurons = (1000, 500)
n_layers = (3, 2, 1)
batch_size = (128,)
n_epochs = 150

lr=0.001

lr_drop = 25

decay = ('cyclic','none')#, 'exp', 'none')

######################################################################################################################

opt = (keras.optimizers.adam(lr=lr),)#, keras.optimizers.nadam(lr=lr))
       #keras.optimizers.SGD(lr=lr, decay=1e-7, momentum=0.9, nesterov=True))



run_comb = list(itertools.product(n_neurons, n_layers, batch_size, opt, decay))

datagen_train= ImageDataGenerator(#featurewise_std_normalization=True,
    #featurewise_center=True
    #samplewise_center=True,
    #samplewise_std_normalization=True
    # rotation_range=15,
    # width_shift_range=0.15,
    # height_shift_range=0.15,
    # shear_range=0.2,
    # channel_shift_range=0.2,
    # fill_mode='nearest'
    # horizontal_flip=False,
    # vertical_flip=False,
    # data_format='channels_last'
)




datagen_train.fit(x_train)
datagen_test = datagen_train

def lr_scheduler(epoch):
    return lr * (0.5 ** (epoch // lr_drop))


run = 0

for v in itertools.product(n_neurons, n_layers, batch_size, opt, decay):
    print('\nStarting run %d/%d' % (run + 1, run_comb.__len__()))
    log_dir = os.path.join("./tensorboard", "run_%d" % run)

    t0 = time.time()
    model = keras.Sequential()

    model.add(InputLayer(input_shape=(*input_size,)))
    model.add(Flatten())
    model.add(BatchNormalization())

    for i in range(v[1]):
        model.add(Dense(v[0]))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        if v[0] is 1000:
            model.add(Dropout((0.5)))
        else:
            model.add(Dropout((0.3)))



    model.add(Dense(output_size))
    model.add(Activation('softmax'))

    print(model.summary())

    tensorboard = TensorBoard(log_dir=log_dir)

    model.compile(loss='categorical_crossentropy',
                  optimizer=v[3],
                  metrics=['accuracy'])

    if v[4] is 'cyclic':

        keras.callbacks.K.set_value(model.optimizer.lr, lr)

        clr = CyclicLR(base_lr=1e-4, max_lr=lr, step_size=int((len(x_train) // v[2]) * 8), mode='triangular')

        model.fit_generator(datagen_train.flow(x_train, y_train, batch_size=v[2]),
                  epochs=n_epochs,
                  validation_data=datagen_test.flow(x_test, y_test),
                  shuffle=True, callbacks=[tensorboard, clr],verbose=0)

    elif v[4] is 'exp':

        keras.callbacks.K.set_value(model.optimizer.lr, lr)

        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        model.fit_generator(datagen_train.flow(x_train, y_train, batch_size=v[2]),
                  epochs=n_epochs,
                  validation_data=datagen_test.flow(x_test, y_test),
                  shuffle=True, callbacks=[tensorboard, reduce_lr],verbose=0)

    elif v[4] is 'none':

        keras.callbacks.K.set_value(model.optimizer.lr, lr)

        model.fit_generator(datagen_train.flow(x_train, y_train, batch_size=v[2]),

                  epochs=n_epochs,
                  validation_data=datagen_test.flow(x_test, y_test),
                  shuffle=True, callbacks=[tensorboard],verbose=0)

    with open(log_dir + "/hyperpars.txt", "w") as text_file:
        print("n_neurons__n_layers__batch_size__opt__decay\n", file=text_file)
        print(v[0], file=text_file)
        print(v[1], file=text_file)
        print(v[2], file=text_file)
        print(v[3], file=text_file)
        print(v[3].get_config(), file=text_file)
        print(v[4], file=text_file)
        print(model.get_config(), file=text_file)

    del model


    run += 1






