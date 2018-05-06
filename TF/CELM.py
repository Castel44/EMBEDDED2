import keras
from keras.datasets import mnist
from TF.elm import elm
import tensorflow as tf
import os
import itertools
import numpy as np
import random
from sklearn.preprocessing import StandardScaler

######################################################################################################################
savedir = os.getcwd() + '/elm_tf_test/'
# Get dataset
print('MNIST DATASET')
train, test = mnist.load_data()
x_train, y_train = train
x_test, y_test = test
del train, test

x_train = x_train.reshape(-1, 28 * 28).astype('float32')
x_test = x_test.reshape(-1, 28 * 28).astype('float32')
'''
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)'''

# Hyperparameters
input_size = x_train.shape[1]
out_class = len(np.unique(y_test))
n_neurons = 1024
batch_size = 1024
norm = 10**1

print('CELM')
W = []  #W.shape = [x_train.shape[1], n_neurons]
B = []  #B.shape = [n_neurons]
i = 0
flag = 1
for i in range(n_neurons):
    # Random choose two classes
    cls = np.random.choice(out_class, 2, replace=False)
    # check for biggest class??
    # Pick two random elements from classes
    x_c1 = np.asarray(random.sample(list(x_train[y_train == cls[0]]), 1)).reshape(28*28)
    x_c2 = np.asarray(random.sample(list(x_train[y_train == cls[1]]), 1)).reshape(28*28)
    W.append(x_c2 - x_c1)
    NormW = np.dot(W[i], W[i])
    if NormW < 1 / (flag * out_class):
        flag += 1
        continue
    W[i] = 2*W[i] / NormW
    B.append((x_c2.dot(x_c2) - x_c1.dot(x_c1)) / NormW)
    #if i % 100 == 0: print('Processing W and B: %d' %i)
    i += 1

W = np.asarray(W, dtype='float32')
B = np.asarray(B, dtype='float32')

print('Apply one hot encoding to labels')
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)


print('Building model ELM, norm = %f' %norm)
model = elm(input_size, out_class, l2norm=norm, type='c')
init_w = tf.get_variable(name='init_w', initializer=np.transpose(W))
init_b = tf.get_variable(name='init_b', initializer=B)
model.add_layer(n_neurons, w_init=init_w, b_init=init_b)
model.compile()
train_acc = model.train(x_train, y_train, batch_size=batch_size)
test_acc = model.evaluate(x_test, y_test)
print('Test accuracy: ', test_acc)
B = model.get_B()
Hw, Hb = model.get_Hw_Hb()
y_out = model.iter_predict(x_test, y_test)
