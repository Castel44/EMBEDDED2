# general imports
import tensorflow as tf


# importing keras dataset
from keras.datasets import cifar10 # standard cifar-10 plain dataset, not pre-scaled


# unpack train and test set
train_set , test_set = cifar10.load_data()
x_train, y_train = train_set
x_test, y_test = test_set

import numpy as np
x_train = np.ndarray.astype(x_train, 'float32')

from tensorflow.contrib.data import Dataset, Iterator

# first the data which has to be processed must be converted in a tf dataset object
training_data = Dataset.from_tensor_slices((x_train,y_train))

def featurewise_mean(image):
    mean = tf.reduce_mean(image , axis=(0,1,2), keep_dims=False)
    #mean = tf.div(tf.reduce_sum(image, [0,1,2], keep_dims=False), 50000*32*32)
    broadcast_shape = [1, 1, 1]
    broadcast_shape[2] = image.shape[3]
    mean = tf.reshape(mean, broadcast_shape)
    image = tf.subtract(image, mean)
    return image,mean


# zca whitening in tensorflow
def tfZCA_fit(image, epsilon=1e-6):
    image,mean = featurewise_mean(image)
    flat = tf.reshape(image, (-1, image.shape[1]*image.shape[2]*image.shape[3]))
    sigma = tf.div(tf.einsum('ij,jk->ik', tf.transpose(flat), flat), tf.cast(flat.shape[0],'float32'))
    s, u, _ = tf.svd(sigma)
    s_inv = tf.div(1.,(tf.sqrt(tf.add(s[tf.newaxis],epsilon))))
    principal_components = tf.einsum('ij,jk->ik', tf.multiply(u, s_inv), tf.transpose(u))
    return principal_components,sigma,flat,mean,image


def tfZCA_compute(image, principal_components):
    flatx = tf.reshape(image, (-1, 32 * 32 * 3))
    whitex = tf.einsum('ij,jk->ik', flatx , principal_components)
    x = tf.reshape(whitex, [-1, 32, 32, 3])
    return x

# using Dataset
# An iterator object will be created to GET BATCHES OF IMAGES AND NOT SINGLE IMAGES
batch_size = 2000

batch = training_data.batch(batch_size)

iterator = batch.make_one_shot_iterator()

next_element = iterator.get_next()


training_init_op = iterator.make_initializer(batch)

fit_input = tf.placeholder(dtype='float32', shape=[50000,32,32,3])





#resized = tf.cast(next_element[0],'float32')
#resized = tf.image.resize_nearest_neighbor(tf.cast(next_element[0],'float32'), [20,20])
#resized = tf.map_fn(lambda frame: tf.image.resize_nearest_neighbor(frame, [28,28]) , tf.cast(next_element[0],'float32') )
#global_contrast_norm = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), resized)
#rand_flipped = tf.map_fn(lambda frame: tf.image.random_flip_left_right(frame), global_contrast_norm)

#whitened = tf.map_fn(lambda frame: tfZCA_compute( frame, princ_comp), tf.cast(next_element[0],'float32'))
whitened = tfZCA_compute(next_element[0], princ_comp)


import time

t0 = time.time()

with tf.Session() as sess:
    # initialize the iterator on the training data
    sess.run(training_init_op)
    princ_comp, sigma, flat, mean, image = sess.run([princ_comp, sigma,flat,mean,image], feed_dict={fit_input: x_train})

    # get each element of the training dataset until the end is reached
    while True:
        try:
            std_image, y = sess.run([whitened, next_element[1]])
            # processing goes here




        except tf.errors.OutOfRangeError:
            print("End of training dataset.")
            break

print('Total time employed to process dataset: %.5f seconds' % (time.time() - t0))

