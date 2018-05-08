import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.callbacks import TensorBoard
import keras

from keras.layers import InputLayer, Dense, Dropout, Activation, Flatten
from lr_sched import CyclicLR

import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons
from sklearn.datasets import load_iris, fetch_olivetti_faces, load_breast_cancer
from synthetic_data import spirals
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import time
import itertools
import os


def prepare_data(X, Y):
    """Scale data and apply one hot encoding to labels"""
    num_class = len(np.unique(Y))
    x_train, x_test, y_train, y_test = train_test_split(X.astype('float32'),
                                                        Y.astype('float32'), shuffle=True,
                                                        test_size=0.2)
    print('Dataset info:')
    print('x_train shape ', x_train.shape)
    print('y_train shape ', y_train.shape)
    print('x_test shape ', x_test.shape)
    print('y_test shape ', y_test.shape)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    y_train = to_categorical(y_train, num_classes=num_class)
    y_test = to_categorical(y_test, num_classes=num_class)

    return x_train, y_train, x_test, y_test, num_class


######################################################################################################################
# Get dataset
# olivetti = fetch_olivetti_faces()
# x_train, y_train, x_test, y_test, num_class = prepare_data(olivetti.data, olivetti.target)

# data, target = load_breast_cancer(return_X_y=True)
data, target = spirals(5000, noise=0.1)

x_train, y_train, x_test, y_test, num_class = prepare_data(data, target)

# Hyperparameters

input_size = x_train.shape[1]
output_size = num_class
n_neurons = (50 ,25, 100)
n_layers = (1, 2, 3)
batch_size = (128, 64, 32)
n_epochs = 400

lr_drop = 10
gamma = 20

######################################################################################################################
start = time.time()

run_time = []
train_acc = []
test_acc = []
run = 0

lr = 0.01

opt = (keras.optimizers.adam(lr=lr), keras.optimizers.nadam(lr=lr),
       keras.optimizers.SGD(lr=lr, decay=1e-7, momentum=0.9, nesterov=True), keras.optimizers.SGD(lr=lr))

decay = ('cyclic', 'exp', 'none')

run_comb = list(itertools.product(n_neurons, n_layers, batch_size, opt, decay))


def lr_scheduler(epoch):
    return lr * (0.5 ** (epoch // lr_drop))


for v in itertools.product(n_neurons, n_layers, batch_size, opt, decay):
    print('\nStarting run %d/%d' % (run + 1, run_comb.__len__()))
    log_dir = os.path.join("./tensorboard", "run_%d" % run)

    t0 = time.time()
    model = Sequential()

    model.add(InputLayer(input_shape=(input_size,)))

    for i in range(v[1]):
        model.add(Dense(v[0]))
        model.add(Activation('relu'))

    model.add(Dense(output_size))
    model.add(Activation('softmax'))

    print(model.summary())

    tensorboard = TensorBoard(log_dir=log_dir)

    model.compile(loss='categorical_crossentropy',
                  optimizer=v[3],
                  metrics=['accuracy'])

    if v[4] is 'cyclic':

        keras.callbacks.K.set_value(model.optimizer.lr, lr)

        clr = CyclicLR(base_lr=lr * 1e-4, max_lr=lr * 5, step_size=int((len(x_train) // v[2]) * 8), mode='triangular')

        model.fit(x_train, y_train,
                  batch_size=v[2],
                  epochs=n_epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True, callbacks=[tensorboard, clr],verbose=1)

    elif v[4] is 'exp':

        keras.callbacks.K.set_value(model.optimizer.lr, lr)

        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        model.fit(x_train, y_train,
                  batch_size=v[2],
                  epochs=n_epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True, callbacks=[tensorboard, reduce_lr],verbose=1)

    elif v[4] is 'none':

        keras.callbacks.K.set_value(model.optimizer.lr, lr)

        model.fit(x_train, y_train,
                  batch_size=v[2],
                  epochs=n_epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True, callbacks=[tensorboard],verbose=1)

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
    run_time.append(time.time() - t0)
    print('Run time: ', run_time[run])
    run += 1

print('\nDone training!')
print('Total time: ', time.time() - start)
