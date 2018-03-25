import h5py as h5

from keras.datasets import cifar10

from elm import elm

import tensorflow as tf

import os

import numpy as np

input_size = 32*32*3
output_size = 10
n_neurons= 8196


ortho_w = tf.orthogonal_initializer()
uni_b= tf.uniform_unit_scaling_initializer()

init_w = tf.get_variable(name='init_w',shape=[input_size,n_neurons], initializer=ortho_w)
init_b = tf.get_variable(name='init_b', shape=[n_neurons], initializer=uni_b)

savedir = os.getcwd() + '/elm_tf_test'

elm2 = elm(input_size,output_size, savedir, name='elm2', l2norm=100)

elm2.add_layer(n_neurons, activation=tf.nn.relu)
elm2.add_layer(n_neurons, activation=tf.nn.relu)

elm2.compile()

train , test  = cifar10.load_data()

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

y_pred = []

while True:
    try: # si può scrivere anche su file qui
        y_pred.append(elm2.iter_predict(iter))

    except tf.errors.OutOfRangeError:
        break

Hw1, Hb1 = elm2.get_Hw_Hb(layer_number=0)
B = elm2.get_B()

del elm2

from keras.layers import Dense, Dropout, Activation, InputLayer, Flatten,BatchNormalization
from keras.callbacks import TensorBoard


model = keras.Sequential()
model.add(InputLayer(input_shape=(input_size,)))
model.add(Dense(n_neurons,kernel_initializer=tf.constant_initializer(Hw1), bias_initializer=tf.constant_initializer(Hb1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
#model.add(Dropout(0.75))
model.add(Dense(10))#,kernel_initializer=tf.constant_initializer(B),use_bias=False))
model.add(Activation('softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy',
                optimizer=keras.optimizers.nadam(lr=1e-4),
                metrics=['accuracy'])

tensorboard = TensorBoard(log_dir=savedir +"/run0", histogram_freq=0,
                              write_graph=False, write_images=False)

model.fit(x_train, y_train,
            batch_size=64,
            epochs=100,
            validation_data=(x_test, y_test),
            shuffle=True, callbacks=[tensorboard])







