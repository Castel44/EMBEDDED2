import numpy as np
from TF.elm import elm
import tensorflow as tf
import itertools
import keras
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Activation, InputLayer, Flatten
from keras.callbacks import TensorBoard
from keras.datasets import cifar10
from keras.utils import to_categorical as OneHot
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
'''
print("Loading Dataset: CIFAR10")
# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = OneHot(y_train, 10)
y_test = OneHot(y_test, 10)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print('Reshape data, X.shape [num_instance, features]')
x_train = x_train.reshape(
    (len(x_train), x_train.shape[1] * x_train.shape[2] * x_train.shape[3]))
x_test = x_test.reshape((len(x_test), x_test.shape[1] * x_test.shape[2] * x_test.shape[3]))

print('Scale data, mean= 0, std= 1')
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
'''

# Get dataset
print('MNIST DATASET')
train, test = mnist.load_data()
x_train,y_train = train
x_test, y_test = test
del train, test
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

######################################################################################################################
# Hyperparameters
savedir = os.getcwd() + '/cifar10_jumpstart/'
input_size = x_train.shape[1]
output_size = 10
n_neurons = 2048
batch_size = 1000
norm = (10**2, )
ortho_w = tf.orthogonal_initializer()
unit_b = tf.uniform_unit_scaling_initializer()
init = ((ortho_w, unit_b),)

train_acc = []
test_acc = []
run = 0
run_comb = list(itertools.product(init, norm))
for v in itertools.product(init, norm):
    print('\nStarting run %d/%d' % (run + 1, run_comb.__len__()))
    model = elm(input_size, output_size, n_neurons, w_initializer=v[0][0], b_initializer=v[0][1],
                l2norm=v[1], batch_size=batch_size)
    train_acc.append(model.train(x_train, y_train))
    test_acc.append(model.evaluate(x_test, y_test))
    print('Test accuracy: ', test_acc[run])
    Hw, Hb = model.get_Hw_Hb()
    B = model.get_B()
    del model
    run += 1

print('Done training!')
# os.system('tensorboard --logdir=%s' % savedir)

model = keras.Sequential()
model.add(InputLayer(input_shape=(input_size,)))
model.add(Dense(n_neurons,kernel_initializer=tf.constant_initializer(Hw),bias_initializer=tf.constant_initializer(Hb)))
model.add(Activation('relu'))
#model.add(Dropout(0.75))
model.add(Dense(10,kernel_initializer=tf.constant_initializer(B),use_bias=False))
model.add(Activation('softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy',
                optimizer=keras.optimizers.nadam(lr=0.0001),
                metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')

model.fit(x_train, y_train,
            batch_size=100,
            epochs=200,
            validation_data=(x_test, y_test),
            shuffle=True, callbacks=[early_stopping])

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])


