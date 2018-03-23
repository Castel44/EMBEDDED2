import tensorflow as tf

from datetime import datetime
import time
import numpy as np


class elm(object):

    def __init__(
            self,
            input_size,
            output_size,
            n_neurons,
            savedir,
            activation=tf.sigmoid,
            type='c',
            name="",
            w_initializer='default',
            b_initializer='default',
            l2norm=None,
            batch_size=1000
    ):

        self.input_size = input_size
        self.output_size = output_size
        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.w_initializer = w_initializer
        self.b_initializer = b_initializer
        self.savedir = savedir
        self.activation = activation
        self.name = name
        self.type = type
        self.metric = None
        self.HH = None
        self.HT = None

        if l2norm is None:
            # TODO: do with tf precision
            l2norm = 50 * np.finfo(np.float32).eps
        self.l2norm = l2norm

        # start tensorflow session
        self.sess = tf.Session()
        # Build TensorFlow graph
        # define graph inputs
        with tf.name_scope("input"):
            self.x = tf.placeholder(dtype='float32', shape=[None, input_size], name='x')
            self.y = tf.placeholder(dtype='float32', shape=[None, output_size], name='labels')

        with tf.name_scope("hidden_layer"):
            if self.w_initializer is 'default' and self.b_initializer is 'default':
                init_w = tf.random_normal_initializer(stddev=tf.div(3.0, tf.sqrt(tf.cast(input_size, tf.float32))))
                init_b = tf.random_normal_initializer()
            else:
                print("Using custom inizialization for ELM: %s" % self.name)
                init_w = w_initializer
                init_b = b_initializer

                assert self.w_initializer or self.b_initializer is not 'default', "Both w_initializer and b_initializer " \
                                                                                  "should be provided when using custom initialization"

            self.Hw = tf.get_variable('Hw', shape=[input_size, n_neurons], dtype=tf.float32, initializer=init_w,
                                      trainable=False)
            self.Hb = tf.get_variable('Hb', shape=[n_neurons], dtype=tf.float32, initializer=init_b, trainable=False)
            tf.get_variable_scope().reuse_variables()

            self.H = self.activation(tf.matmul(self.x, self.Hw) + self.Hb)

        with tf.name_scope('output_layer'):
            self.B = tf.Variable(tf.zeros(shape=[n_neurons, output_size]), dtype=tf.float32, name='B')
            self.y_out = tf.matmul(self.H, self.B)

        if self.type is 'c':
            with tf.name_scope("accuracy"):
                self.correct_prediction = tf.equal(tf.argmax(self.y_out, 1), tf.argmax(self.y, 1))
                self.metric = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        elif self.type is 'r':  # regression
            with tf.name_scope("mean_squared_error"):
                # nb possible numerical instability
                # https://stackoverflow.com/questions/41338509/tensorflow-mean-squared-error-loss-function
                # TODO
                self.metric = tf.reduce_mean(tf.squared_difference(self.y_out, self.y, name='mse'))
        else:
            raise ValueError("Invalid argument for type")

        # initialization
        self.sess.run([self.Hw.initializer, self.Hb.initializer])
        print("Network parameters have been initialized")

        self.writer = tf.summary.FileWriter(self.savedir + self.name)
        self.writer.add_graph(self.sess.graph)

    def evaluate(self, dataset):
        # create tensorflow iterator
        batched_dataset = dataset.batch(batch_size=self.batch_size)
        iterator = batched_dataset.make_initializable_iterator()
        next_batch = iterator.get_next()
        self.sess.run(iterator.initializer)
        metric_vect = []
        while True:
            try:
                x_batch, y_batch = self.sess.run(next_batch)
                metric_vect.append(self.sess.run(self.metric, feed_dict={self.x: x_batch, self.y: y_batch}))
            except tf.errors.OutOfRangeError:
                break
        return np.mean(
            metric_vect)  # TODO self.y_out return ? maybe predic retuns only y_out and then another method computer accuracy

    def train(self, dataset):
        # define training structure
        with tf.name_scope("training"):
            # initialization and training graph definition
            self.HH = tf.Variable(tf.multiply(tf.eye(self.n_neurons, dtype=tf.float32), self.l2norm), name='HH')
            self.HT = tf.Variable(tf.zeros([self.n_neurons, self.output_size]), name='HT')

            train_op = tf.group(
                tf.assign_add(self.HH, tf.matmul(self.H, self.H, transpose_a=True)),
                tf.assign_add(self.HT, tf.matmul(self.H, self.y, transpose_a=True))
            )
            # TODO: check for correct matrix inversion HH
            B_op = tf.assign(self.B, tf.matmul(tf.matrix_inverse(self.HH), self.HT))

        # add to graph
        self.writer.add_graph(self.sess.graph)

        # Initialize a saver to store model checkpoints
        # saver = tf.train.Saver()

        # create tensorflow iterator
        batched_dataset = dataset.batch(batch_size=self.batch_size)
        iterator = batched_dataset.make_initializable_iterator()
        next_batch = iterator.get_next()

        nb = int(np.ceil(dataset._tensors[1]._shape_as_list()[0] / self.batch_size))  # ceil ?? #TODO

        # initialize variables
        self.sess.run([self.HH.initializer, self.HT.initializer])
        self.sess.run(iterator.initializer)

        t0 = time.time()
        print("{} Start training...".format(datetime.now()))
        print("{} Open tensorboard at --logdir={}".format(datetime.now(), self.savedir))

        batch = 1
        while True:
            try:
                start = time.time()
                # get next batch of data
                x_batch, y_batch = self.sess.run(next_batch)
                # Run the training op
                self.sess.run(train_op, feed_dict={self.x: x_batch, self.y: y_batch})
                eta = (time.time() - start) * (nb - batch)
                eta = '%d:%02d' % (eta // 60, eta % 60)
                print("{}/{} ETA:{}".format(batch, nb, eta))
                batch += 1
            except tf.errors.OutOfRangeError:
                break
        # TODO: check if HH is invertible
        self.sess.run(B_op)
        print("Training of ELM {} ended in {}:{:5f}".format(self.name, ((time.time() - t0) // 60),
                                                            ((time.time() - t0) % 60)))
        print("#" * 100)

        # Compute train accuracy
        train_metric = self.evaluate(dataset)
        if self.type is 'c':
            print('Train accuracy: ', train_metric)
        else:  # regression
            print('Train MSE: ', train_metric)

    def get_Hw_Hb(self):
        Hw = self.Hw.eval()  # get hidden layer weights matrix
        Hb = self.Hb.eval()  # get hidden layer biases
        return Hw, Hb

    def get_B(self):
        return self.B.eval()

    def get_HH(self):
        return self.HH.eval()

    def __del__(self):
        self.sess.close()
        tf.reset_default_graph()
        print("TensorFlow graph resetted")
