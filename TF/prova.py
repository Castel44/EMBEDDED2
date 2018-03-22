import h5py as h5

from keras.datasets import mnist

from elm import elm

import tensorflow as tf

import os

input_size = 784
output_size = 10
n_neurons= 2048

ortho_w = tf.orthogonal_initializer()
unit_b = tf.uniform_unit_scaling_initializer()

init_w = tf.get_variable(name='init_w',shape=[input_size,n_neurons], initializer=ortho_w)#tf.Variable(initial_value=ortho_w.__call__(shape=[input_size,n_neurons])) # correct way to use initializers
init_b = tf.Variable(initial_value=unit_b.__call__(shape=[n_neurons]))


savedir = os.getcwd() + '/elm_tf_test'


elm1 = elm(input_size,output_size,n_neurons, savedir, type='r')


hw = elm1.Hw


elm2 = elm(input_size,output_size,n_neurons, savedir, w_initializer= init_w, b_initializer=init_b, )

train , test  = mnist.load_data(savedir + "/mnist.txt")

x_train,y_train = train
x_test, y_test = test

import keras

y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

x_train = x_train.reshape(-1,28*28)
x_test = x_test.reshape(-1,28*28)


dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))


elm2.train(dataset,batch_size=2000)
print(elm2.batch_predict(tf.data.Dataset.from_tensor_slices((x_test, y_test))))


elm1.train(dataset,batch_size=2000)
print(elm1.batch_predict(tf.data.Dataset.from_tensor_slices((x_test, y_test))))
