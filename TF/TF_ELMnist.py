import os
import numpy as np
import tensorflow as tf
import time
import itertools
import keras
from keras.datasets import mnist


def pinv(A, b, reltol=1e-6):
    # Compute the SVD of the input matrix A
    s, u, v = tf.svd(A)

    # Invert s, clear entries lower than reltol*s[0].
    atol = tf.reduce_max(s) * reltol
    s = tf.boolean_mask(s, s > atol)
    s_inv = tf.diag(tf.concat([1. / s, tf.zeros([tf.size(b) - tf.size(s)])], 0))

    # Compute v * s_inv * u_t * b from the left to avoid forming large intermediate matrices.
    return tf.matmul(v, tf.matmul(s_inv, tf.matmul(u, tf.reshape(b, [-1, 1]), transpose_a=True)))


# TODO: make predict in batch_way
# TODO: fix correct histogram and variable TensorBoard
# TODO: better hypar management, do dict or smth
def ELM_classificator(size_in, size_out, n_neuron, batch_size, norm):
    tf.reset_default_graph()
    sess = tf.Session()

    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.float32, shape=[None, size_in], name="x")
        y = tf.placeholder(tf.float32, shape=[None, size_out], name="labels")

    with tf.name_scope('hidden_layer'):
        w = tf.Variable(tf.random_normal(shape=[size_in, n_neuron], stddev=1, dtype=tf.float32), trainable=False,
                        name='w')
        # w = tf.get_variable('w', shape=[size_in, n_neuron],dtype=tf.float32, initializer=tf.orthogonal_initializer(), trainable=None)
        b = tf.Variable(tf.random_normal(shape=[n_neuron], stddev=1, dtype=tf.float32), trainable=False, name='b')
        H = tf.sigmoid(tf.matmul(x, w) + b)  # H idden reprs

    with tf.name_scope('train'):
        B = tf.Variable(tf.zeros([n_neuron, size_out], dtype=tf.float32), name='B')
        HH = tf.Variable(tf.multiply(tf.eye(n_neuron, dtype=tf.float32), 10 ** norm), name='HH')
        HT = tf.Variable(tf.zeros([n_neuron, size_out], dtype=tf.float32), name='HT')
        HH_HT_op = tf.group(
            HH.assign_add(tf.matmul(H, H, transpose_a=True)),
            HT.assign_add(tf.matmul(H, y, transpose_a=True))
        )
        B_op = B.assign(tf.matmul(tf.matrix_inverse(HH), HT))

    with tf.name_scope('output_layer'):
        y_proba = tf.matmul(H, B)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(y_proba, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())
    # saver = tf.train.Saver()
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
        sess.run(HH_HT_op, feed_dict={x: x_batch, y: y_batch})
    sess.run(B_op)
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
neuron_number = (5000,)
norm = (-3,)

train_acc = []
test_acc = []
run = 0
run_comb = list(itertools.product(neuron_number, norm))
for v in itertools.product(neuron_number, norm):
    print('Starting run %d/%d' % (run + 1, run_comb.__len__()))
    train, test = ELM_classificator(x_train.shape[1], 10, v[0], batch_size, v[1])
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
