import h5py as h5
import keras
from keras.datasets import mnist
from TF.elm import elm
import tensorflow as tf
import os

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
x_train = x_train.reshape(-1,28*28)
x_test = x_test.reshape(-1,28*28)

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

# Hyperparameters
input_size = x_train.shape[1]
output_size = 10
n_neurons = 5000
batch_size = 5000
norm = 10 ** -3

elm1 = elm(input_size, output_size, n_neurons, savedir, name='default', type='c', l2norm=None, batch_size=batch_size)
elm1.train(dataset)
print('Default init: ', elm1.evaluate(tf.data.Dataset.from_tensor_slices((x_test, y_test))))
elm1.__del__()

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ortho_w = tf.orthogonal_initializer()
unit_b = tf.uniform_unit_scaling_initializer()
elm2 = elm(input_size, output_size, n_neurons, savedir, name='orth', w_initializer=ortho_w, b_initializer=unit_b,
           l2norm=None, batch_size=batch_size)
elm2.train(dataset)
print('Ortho init: ', elm2.evaluate(tf.data.Dataset.from_tensor_slices((x_test, y_test))))

os.system('tensorboard --logdir=%s' % savedir)
