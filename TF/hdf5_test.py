import h5py

from keras.datasets import cifar10

from elm import elm

import tensorflow as tf

import os

import numpy as np

input_size = 32*32*3
output_size = 10
n_neurons= 500


ortho_w = tf.orthogonal_initializer()
uni_b= tf.uniform_unit_scaling_initializer()

init_w = tf.get_variable(name='init_w',shape=[input_size,n_neurons], initializer=ortho_w)
init_b = tf.get_variable(name='init_b', shape=[n_neurons], initializer=uni_b)

savedir = os.getcwd() + '/elm_tf_test'

elm2 = elm(input_size,output_size, savedir, name='elm2', l2norm=100)

elm2.add_layer(n_neurons, activation=tf.nn.relu)
elm2.add_layer(n_neurons, activation=tf.nn.relu)

elm2.compile()

train ,test = cifar10.load_data()

x_train,y_train = train
x_test, y_test = test

import keras

y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

x_train = x_train.reshape(-1,input_size)
x_test = x_test.reshape(-1,input_size)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

elm2.train(x_train, y_train,batch_size=1000)
acc = elm2.evaluate(x_test,y_test)
print(acc)

iter = elm2.get_iterator(x_test,y_test)

hdf5file = os.getcwd() +"/elm2.hdf5"

dataname = 'elm2'

elm2.iter_predict(x_train,y_train, dataname=dataname, filepath=hdf5file)

y_pred_list = elm2.iter_predict(x_train,y_train)

y_pred_h5 = []

with h5py.File(hdf5file, "r") as f:
    data_size = f[dataname].shape[0]
    y_pred_h5 = f[dataname][0:data_size]

print(np.array_equal(y_pred_h5,y_pred_list))



