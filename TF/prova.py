import h5py as h5

from keras.datasets import mnist

from elm import elm

import tensorflow as tf

import os

input_size = 784
output_size = 10
n_neurons=5000


ortho_w = tf.orthogonal_initializer()

init_w = tf.get_variable(name='init_w',shape=[input_size,n_neurons], initializer=ortho_w)#tf.Variable(initial_value=ortho_w.__call__(shape=[input_size,n_neurons])) # correct way to use initializers



savedir = os.getcwd() + '/elm_tf_test'

elm2 = elm(input_size,output_size, savedir, name='elm2' )

elm2.add_layer(5000, w_init=init_w, b_init=None)


elm2.compile()

train , test  = mnist.load_data(savedir + "/mnist.txt")

x_train,y_train = train
x_test, y_test = test

import keras

y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

x_train = x_train.reshape(-1,28*28)
x_test = x_test.reshape(-1,28*28)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

elm2.train(x_train, y_train,batch_size=2000)
acc = elm2.evaluate(x_test,y_test)
print(acc)

#del elm2

elm1 = elm(input_size,output_size, savedir, name='elm1' )

elm1.add_layer(5000, w_init=init_w, b_init=None)


elm1.compile()

elm1.train(x_train, y_train,batch_size=2000)






