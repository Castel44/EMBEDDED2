import os
import numpy as np
import tensorflow as tf
import sys
from keras.datasets import mnist
import keras

LOGDIR = '/tmp/TF_ELM_mnist/'

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
print(x_train.shape)

n_neuron = 2048
norm = 10 ** 6
batch_size = 5000

tf.reset_default_graph()

with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32, shape=[None, 28 * 28], name="x")
    y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")

with tf.name_scope('hidd'):
    w = tf.Variable(tf.random_normal(shape=[28 * 28, n_neuron], stddev=1), trainable=False)
    b = tf.Variable(tf.random_normal(shape=[n_neuron], stddev=1), trainable=False)
    H = tf.sigmoid(tf.matmul(x, w) + b)  # H idden reprs
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", H)

with tf.name_scope('train'):
    I = tf.eye(n_neuron, dtype=tf.float32)
    HH = tf.Variable(tf.add(tf.zeros(n_neuron), tf.div(I, norm)), name='HH')
    HT = tf.Variable(tf.zeros([n_neuron, 10]), name='HT')
    HH_op = tf.assign(HH, tf.add(HH, tf.matmul(H, H, transpose_a=True)))
    HT_op = tf.assign(HT, tf.add(HT, tf.matmul(H, y, transpose_a=True)))
    # Does need assign for B?
    B = tf.matmul(tf.matrix_inverse(HH), HT)
    tf.summary.histogram("B", B)
    tf.summary.histogram("HH", HH)
    tf.summary.histogram("HT", HT)

with tf.name_scope('output_layer'):
    y_proba = tf.matmul(H, B)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y_proba, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    summ = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOGDIR)
    writer.add_graph(sess.graph)

    print("initialized")
    # TODO; better, ceil
    nb = x_train.shape[0] // batch_size

    for i in range(nb):
        print('Processing batch %d/%d' % (i + 1, nb))
        x_batch = x_train[i * batch_size:((i + 1) * batch_size)].astype('float32')
        y_batch = y_train[i * batch_size:((i + 1) * batch_size)].astype('float32')
        [_, _, s] = sess.run([HH_op, HT_op, summ], feed_dict={x: x_batch, y: y_batch})
        writer.add_summary(s, i)

    acc_train = accuracy.eval(feed_dict={x: x_train, y: y_train})
    acc_test = accuracy.eval(feed_dict={x: x_test, y: y_test})

print(acc_train)
print(acc_test)

os.system('tensorboard --logdir=%s' % LOGDIR)
