import keras
from keras.datasets import mnist
from keras.datasets import cifar10
from TF.elm import elm as elm
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import os
import itertools
import numpy as np
import time
from datetime import datetime

def create_iterator(x, y, batch_size=1000):
    print('Creating dataset iterator')
    t = time.time()
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    batched_dataset = dataset.batch(batch_size=batch_size)
    iterator = batched_dataset.make_initializable_iterator()
    print('{} Iterator created in {:f}' .format(datetime.now(), time.time()-t))
    return iterator

######################################################################################################################
savedir = os.getcwd() + '/elm_tf_test/'

# Get dataset
'''
print('MNIST DATASET')
train, test = mnist.load_data()
x_train, y_train = train
x_test, y_test = test
del train, test
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)'''


print("Loading Dataset: CIFAR10")
# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)
x_train = x_train.reshape(-1, 32 * 32 * 3).astype('float32')
x_test = x_test.reshape(-1, 32 * 32 * 3).astype('float32')

prescaler = StandardScaler()
x_train = prescaler.fit_transform(x_train.astype('float32'))
x_test = prescaler.transform(x_test.astype('float32'))

# Hyperparameters
input_size = x_train.shape[1]
output_size = 10
n_neurons = 4096
batch_size = 1000
repeate_run = 1
norm = repeate_run*(10**-5, 10**-2, 10**0, 10**3,)
ortho_w = tf.orthogonal_initializer()
uni_b = tf.variance_scaling_initializer(distribution='uniform')
init = (['default', 'default'],)

train_iter = create_iterator(x_train, y_train, batch_size=batch_size)

train_acc = []
test_acc = []
run = 0
run_comb = list(itertools.product(init, norm))
for v in itertools.product(init, norm):
    t = time.time()
    print('\nStarting run %d/%d' % (run + 1, run_comb.__len__()))
    print('Hyperpar: init=', v[0], 'norm=', v[1])
    model = elm(input_size, output_size, l2norm=v[1], type='c')
    if v[0][1] is not 'default':
        with tf.variable_scope('custom_init', reuse=tf.AUTO_REUSE):
            v[0][0] = tf.get_variable(name='init_w', shape=[input_size, n_neurons], initializer=ortho_w)
            v[0][1] = tf.get_variable(name='init_b', shape=[n_neurons], initializer=uni_b)
    model.compile(n_neurons, w_init=v[0][0], b_init=v[0][1])
    train_acc.append(model.train(x_train, y_train, iterator=train_iter, batch_size=batch_size))
    test_acc.append(model.evaluate(x_test, y_test, batch_size=batch_size))
    print('Test accuracy: ', test_acc[run])

    #B = model.get_B()
    #Hw, Hb = model.get_Hw_Hb()
    #y_out = model.iter_predict(x_test, y_test)
    del model
    run += 1
    print('Run time ', time.time() - t)

print('Done training!')
# os.system('tensorboard --logdir=%s' % savedir)

# Searching for best hypar combination
best_net = np.argmax(test_acc)
print('Best net with hypepar:')
print('  -neuron number:', run_comb[best_net][0])
print('  -norm:', run_comb[best_net][1])
print('Best net test accuracy: ', test_acc[best_net])

# mean accuracy
print('Mean acc from %d run: %f' %(run_comb.__len__(), sum(test_acc)/len(test_acc)))

#CIFAR ~43%
