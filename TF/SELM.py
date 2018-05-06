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


print('SELM')
W = []  #W.shape = [x_train.shape[1], n_neurons]
B = np.random.randn(n_neurons) #B.shape = [n_neurons]
for i in range(n_neurons):
    x = np.asarray(random.sample(list(x_train), 1)).reshape(28*28)
    W.append(x/x.dot(x))
    NormW = np.dot(x,x)
    if i % 100 == 0: print('Processing W and B: %d' %i)

W = np.asarray(W, dtype='float32')
B = np.asarray(B, dtype='float32')

# delete 0 value from W
from numpy import linalg as LA
W = W[LA.norm(W, axis=1) == 0]
n_neurons = W.shape[0]

print('Apply one hot encoding to labels')
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)


norm = (10**2,)
init = ([np.transpose(W), B],)

train_acc = []
test_acc = []
run = 0
run_comb = list(itertools.product(init, norm))
for v in itertools.product(init, norm):
    print('\nStarting run %d/%d' % (run + 1, run_comb.__len__()))
    print('Hyperpar: init=', v[0], 'norm=', v[1])
    model = elm(input_size, out_class, l2norm=v[1], type='c')
    if v[0][1] is not 'default':
        init_w = tf.get_variable(name='init_w', initializer=np.transpose(W))
        init_b = tf.get_variable(name='init_b', initializer=B)
    else:
        init_w = init_b = 'default'
    model.add_layer(n_neurons, w_init=init_w, b_init=init_b)
    model.compile()
    train_acc.append(model.train(x_train, y_train, batch_size=batch_size))
    test_acc.append(model.evaluate(x_test, y_test))
    print('Test accuracy: ', test_acc[run])

    B = model.get_B()
    Hw, Hb = model.get_Hw_Hb()
    y_out = model.iter_predict(x_test, y_test)
    del model, init_b, init_w
    run += 1

print('Done training!')
# os.system('tensorboard --logdir=%s' % savedir)

# Searching for best hypar combination
best_net = np.argmax(test_acc)
print('Best net with hypepar:')
print('  -neuron number:', run_comb[best_net][0])
print('  -norm:', run_comb[best_net][1])
print('Best net test accuracy: ', test_acc[best_net])