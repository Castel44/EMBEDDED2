import h5py as h5
import keras
from keras.datasets import mnist
from TF.elm import elm
import tensorflow as tf
import os
import itertools
import numpy as np

######################################################################################################################
savedir = os.getcwd() + '/elm_tf_test/'
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

# dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

# Hyperparameters
input_size = x_train.shape[1]
output_size = 10
n_neurons = 5000
batch_size = 5000
norm = (None, 10 ** -3, 10 ** 0, 10 ** 2, 10 ** 4,)
ortho_w = tf.orthogonal_initializer()
unit_b = tf.uniform_unit_scaling_initializer()
init = ((None, None), (ortho_w, unit_b),)

train_acc = []
test_acc = []
run = 0
run_comb = list(itertools.product(init, norm))
for v in itertools.product(init, norm):
    print('Starting run %d/%d' % (run + 1, run_comb.__len__()))
    model = elm(input_size, output_size, n_neurons, w_initializer=v[0][0], b_initializer=v[0][1],
                l2norm=v[1], batch_size=batch_size)
    train_acc.append(model.train(x_train, y_train))
    test_acc.append(model.evaluate(x_test, y_test))
    print('Test accuracy: ', test_acc[run])
    del model
    run += 1

print('Done training!')
# os.system('tensorboard --logdir=%s' % savedir)

# Searching for best hypar combination
best_net = np.argmax(test_acc)
print('Best net with hypepar:')
print('  -neuron number: ', run_comb[best_net][0])
print('  -norm: 10 **', run_comb[best_net][1])
print('Best net test accuracy: ', test_acc[best_net])
