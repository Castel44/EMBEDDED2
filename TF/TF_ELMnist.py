import os
import numpy as np
import tensorflow as tf
import time
import itertools
import keras
from keras.datasets import mnist


# TODO: make predict in batch_way
# TODO: fix correct histogram and variable TensorBoard
# TODO: better hypar management, do dict or smth
def ELM_classificator(size_in, size_out, n_neuron, batch_size, norm):
    tf.reset_default_graph()
    sess = tf.Session()

    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.float32, shape=[None, size_in * size_in], name="x")
        y = tf.placeholder(tf.float32, shape=[None, size_out], name="labels")

    with tf.name_scope('hidden_layer'):
        w = tf.Variable(tf.random_normal(shape=[size_in * size_in, n_neuron], stddev=1), trainable=False)
        b = tf.Variable(tf.random_normal(shape=[n_neuron], stddev=1), trainable=False)
        H = tf.sigmoid(tf.matmul(x, w) + b)  # H idden reprs

    with tf.name_scope('train'):
        I = tf.eye(n_neuron, dtype=tf.float32)
        HH = tf.Variable(tf.add(tf.zeros(n_neuron), tf.div(I, 10 ** norm)), name='HH')
        HT = tf.Variable(tf.zeros([n_neuron, size_out]), name='HT')
        HH_op = tf.assign(HH, tf.add(HH, tf.matmul(H, H, transpose_a=True)))
        HT_op = tf.assign(HT, tf.add(HT, tf.matmul(H, y, transpose_a=True)))
        B = tf.matmul(tf.matrix_inverse(HH), HT)

    with tf.name_scope('output_layer'):
        y_proba = tf.matmul(H, B)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(y_proba, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())
    hparm = 'n_%d,norm_%d' % (n_neuron, norm)
    writer = tf.summary.FileWriter(LOGDIR + hparm)
    writer.add_graph(sess.graph)

    print("Starting run for %s" % hparm)
    nb = int(np.ceil(x_train.shape[0] / batch_size))
    t0 = time.time()
    for i in range(nb):
        print('Processing batch %d/%d' % (i + 1, nb))
        x_batch = x_train[i * batch_size:((i + 1) * batch_size)].astype('float32')
        y_batch = y_train[i * batch_size:((i + 1) * batch_size)].astype('float32')
        sess.run([HH_op, HT_op], feed_dict={x: x_batch, y: y_batch})

    print('Training done in: ', time.time() - t0)
    acc_train = accuracy.eval(feed_dict={x: x_train, y: y_train}, session=sess)
    print('Train accuracy: ', acc_train)
    acc_test = accuracy.eval(feed_dict={x: x_test, y: y_test}, session=sess)
    print('Test accuracy: ', acc_test)
    print('#' * 100)
    return acc_train, acc_test


######################################################################################################################
LOGDIR = '/tmp/TF_ELM_mnist/'

print('MNIST DATASET')
mnist_data = mnist.load_data()
train_set, test_set = mnist_data
x_train, y_train = train_set
x_test, y_test = test_set

y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)
print('x_train shape: ', x_train.shape)
# Dataset not normalized


# HYPEPARAMETERS
batch_size = 5000
neuron_number = (10000,)
norm = (3,)

train_acc = []
test_acc = []
run = 0
run_comb = list(itertools.product(neuron_number, norm))
for v in itertools.product(neuron_number, norm):
    print('Starting run %d/%d' % (run + 1, run_comb.__len__()))
    train, test = ELM_classificator(28, 10, v[0], batch_size, v[1])
    train_acc.append(train)
    test_acc.append(test)
    run += 1

print('Done training!')
print('Run `tensorboard --logdir=%s` to see the results.' % LOGDIR)
# os.system('tensorboard --logdir=%s' % LOGDIR)

# Searching for best hypar combination
best_net = np.argmax(test_acc)
print('Best net with hypepar:')
print('  -neuron number: ', run_comb[best_net][0])
print('  -norm: 10 **', run_comb[best_net][1])
print('Best net test accuracy: ', test_acc[best_net])
