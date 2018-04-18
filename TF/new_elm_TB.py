from TF.elm_new import elm
import tensorflow as tf
import numpy as np
import keras
from keras.datasets import cifar10
from keras.datasets import mnist
import os
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
import time
import itertools


# Get dataset
'''
train, test = mnist.load_data()
x_train, y_train = train
x_test, y_test = test
del train, test
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28 * 28)'''


print("Loading Dataset: CIFAR10")
# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)
x_train = x_train.reshape(-1, 32,32,3).astype('float32')
x_test = x_test.reshape(-1, 32,32,3).astype('float32')

# Hyperparameters
input_size = 32*32*3
output_size = 10
n_neurons = (4096,)
batch_size = 5000
n_epochs = 1
repeate_run = 1
norm = repeate_run*(None, 10**0, 10**3,)
ortho_w = tf.orthogonal_initializer()
uni_b = tf.variance_scaling_initializer(distribution='uniform')
init = (['default', 'default'], [ortho_w, uni_b],)

# pre-processing pipeline
start = time.time()
datagen = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    #rotation_range=15,
    #width_shift_range=0.15,
    #height_shift_range=0.15,
    #shear_range=0.2,
    #channel_shift_range=0.2,
    #fill_mode='nearest'
    #horizontal_flip=False,
    #vertical_flip=False,
    #data_format='channels_last'
)

datagen.fit(x_train)

def gen():
    n_it = 0
    batches_per_epochs = len(x_train) // batch_size
    for x, y in datagen.flow(x_train, y_train, batch_size=batch_size):
        x = x.reshape(batch_size, x_train.shape[1] * x_train.shape[2] * x_train.shape[3])
        if n_it % 10 == 0:
           print("generator iteration: %d" % n_it)
        yield x, y
        n_it += 1
        if n_it >= batches_per_epochs * n_epochs:
            break


def gen_test():
    n_it = 0
    batches_per_epochs = len(x_test) // batch_size
    for x, y in datagen.flow(x_test, y_test, batch_size=batch_size):
        x = x.reshape(batch_size, x_test.shape[1] * x_test.shape[2] * x_test.shape[3])
        if n_it % 10 == 0:
            print("generator iteration: %d" % n_it)
        yield x, y
        n_it += 1
        if n_it >= batches_per_epochs * n_epochs:
            break


data = tf.data.Dataset.from_generator(generator=gen,
                                      output_shapes=((batch_size, 32 * 32 * 3,), (batch_size, 10,)),
                                      output_types=(tf.float32, tf.float32))

data2 = tf.data.Dataset.from_generator(generator=gen_test,
                                      output_shapes=((batch_size, 32 * 32 * 3,), (batch_size, 10,)),
                                      output_types=(tf.float32, tf.float32))

iterator = data.make_initializable_iterator()
iterator2 = data2.make_initializable_iterator()

train_acc = []
test_acc = []
run = 0
run_comb = list(itertools.product(n_neurons, init, norm))
for v in itertools.product(n_neurons, init, norm):
    print('\nStarting run %d/%d' % (run + 1, run_comb.__len__()))
    print('Hyperpar: neurons= ', v[0], 'init=', v[1], 'norm=', v[2])
    t0 = time.time()
    model = elm(input_size=input_size, output_size=output_size, l2norm=v[2])
    if v[1][1] is not 'default':
        with tf.variable_scope('custom_init', reuse=tf.AUTO_REUSE):
            v[1][0] = tf.get_variable(name='init_w', shape=[input_size, v[0]], initializer=ortho_w)
            v[1][1] = tf.get_variable(name='init_b', shape=[v[0]], initializer=uni_b)
    model.add_layer(v[0], activation=tf.sigmoid, w_init=v[1][0], b_init=v[1][1])
    model.compile()

    model.sess.run([iterator.initializer, iterator2.initializer])

    model.train(iterator, n_batches=n_epochs * (len(x_train) // batch_size))
    test_acc.append(model.evaluate(tf_iterator= iterator2, batch_size=1000))
    print('Test accuracy: ', test_acc[run])

    #B = model.get_B()
    #Hw, Hb = model.get_Hw_Hb()
    #y_out = model.iter_predict(x_test, y_test)
    #tf.reset_default_graph()
    del model
    run += 1
    print('Run time: ', time.time() - t0)

print('\nDone training!')
print('Total time: ', time.time() - start)

# Searching for best hypar combination
best_net = np.argmax(test_acc)
print('Best net with hypepar:')
print('  -neuron number:', run_comb[best_net][0])
print('  -norm:', run_comb[best_net][2])
print('Best net test accuracy: ', test_acc[best_net])

# mean accuracy
# be careful when multi-parameter are chosen, need to decimate A[::10] of repeate_run to obtain correct accuracy for every hyperpar
if repeate_run > 1:
    print('Mean acc from %d run: %f' %(run_comb.__len__(), sum(test_acc)/len(test_acc)))
