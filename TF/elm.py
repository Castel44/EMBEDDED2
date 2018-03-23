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
            activation= tf.sigmoid,
            type = 'c',
            name = "",
            w_initializer = 'default',
            b_initializer = 'default',
            l2norm=0.,

    ):

        self.input_size = input_size
        self.output_size = output_size
        self.n_neurons = n_neurons
        self.l2norm = l2norm
        self.w_initializer = w_initializer
        self.b_initializer = b_initializer
        self.savedir = savedir
        self.activation = activation
        self.name = name
        self.type = type
        self.metric = None
        self.HH = None
        self.HT = None


        # start tensorflow session
        self.sess = tf.Session()

        # define graph inputs
        with tf.name_scope("input"):

            self.x = tf.placeholder(dtype='float32', shape=[None, input_size])
            self.y = tf.placeholder(dtype='float32', shape=[None, output_size])

        with tf.name_scope("hidden_layer"):

            if self.w_initializer is 'default' and self.b_initializer is 'default':


                init_w = tf.random_normal(shape=[self.input_size, self.n_neurons],

                                  stddev=tf.sqrt(tf.div(2.,
                                                        tf.add(tf.cast(self.input_size, 'float32'),
                                                        tf.cast(self.output_size, 'float32')))))

                self.Hw = tf.Variable(init_w)


                init_b = tf.random_normal(shape=[self.n_neurons],

                                          stddev=tf.sqrt(tf.div(2.,
                                                                tf.add(tf.cast(self.input_size, 'float32'),
                                                                tf.cast(self.output_size, 'float32')))))

                self.Hb = tf.Variable(init_b)


            else:

                print("Using custom inizialization for ELM: %s" % self.name)

                self.Hw = w_initializer

                self.Hb = b_initializer

                assert self.Hw or self.Hb is not 'default', "Both w_initializer and b_initializer " \
                                                           "should be provided when using custom initialization"


                assert self.Hw.shape.as_list() is not [input_size, n_neurons], "Invalid shape for hidden layer weights tensor"
                assert self.Hb.shape.as_list() is not [n_neurons], "Invalid shape for hidden layer biases tensor"


            self.H = self.activation(tf.matmul(self.x, self.Hw) + self.Hb)


            with tf.name_scope('output_layer'):

                    self.B = tf.Variable(tf.zeros(shape=[self.n_neurons, self.output_size]), dtype='float32')

                    self.y_out = tf.matmul(self.H, self.B)


        # initialization
        self.sess.run([self.Hw.initializer, self.Hb.initializer])
        print("Network parameters have been initialized")

        self.writer = tf.summary.FileWriter(self.savedir + self.name)
        self.writer.add_graph(self.sess.graph)




    def evaluate(self, dataset, batch_size=10000):

        # create tensorflow iterator
        batched_dataset = dataset.batch(batch_size=batch_size)

        # create tensorflow iterator
        iterator = batched_dataset.make_initializable_iterator()

        next_batch = iterator.get_next()

        self.sess.run(iterator.initializer)

        metric_vect = []

        while True:
            try:
                x_batch, y_batch = self.sess.run(next_batch)

                metric_vect.append(self.sess.run(self.metric,feed_dict={self.x: x_batch, self.y: y_batch}))

            except tf.errors.OutOfRangeError:

                break

        return np.mean(metric_vect) #TODO self.y_out return ? maybe predic retuns only y_out and then another method computer accuracy



    def train(self, dataset, batch_size = 2000):

        # define training structure

        with tf.name_scope("training"):

            # initialization and training graph definition

            self.HH = tf.Variable(tf.multiply(tf.eye(self.n_neurons, dtype=tf.float32),
                                              tf.cast(self.l2norm, tf.float32)),
                                              name='HH')

            self.HT = tf.Variable(tf.zeros([self.n_neurons, self.output_size]), name='HT')


            train_op = tf.group(
             tf.assign_add(self.HH, tf.matmul(self.H, self.H, transpose_a=True)),
             tf.assign_add(self.HT, tf.matmul(self.H, self.y, transpose_a=True))
            )


            B_op = tf.assign(self.B, tf.matmul(tf.matrix_inverse(self.HH), self.HT))



        if self.type is 'c':

            # no need for cost function # TODO

            with tf.name_scope("accuracy"):
                correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_out, 1))
                self.metric = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        elif self.type is 'r': # regression
            with tf.name_scope("mean_squared_error"):
                # nb possible numerical instability
                # https://stackoverflow.com/questions/41338509/tensorflow-mean-squared-error-loss-function
                # TODO

                self.metric = tf.reduce_mean(tf.squared_difference(self.y_out,self.y, name='mse'))

        else:

            raise ValueError("Invalid argument for type")

        # add to graph

        self.writer.add_graph(self.sess.graph)

        # Initialize a saver to store model checkpoints
        saver = tf.train.Saver()

        batched_dataset = dataset.batch(batch_size=batch_size)

        # create tensorflow iterator
        iterator = batched_dataset.make_initializable_iterator()

        next_batch = iterator.get_next()

        nb = int(np.ceil(dataset._tensors[1]._shape_as_list()[0]/batch_size)) # ceil ?? #TODO

        # initialize variables

        self.sess.run([self.HH.initializer, self.HT.initializer])
        self.sess.run(iterator.initializer)

        t0 = time.time()

        print("{} Start training...".format(datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                        self.savedir))

        batch=1
        while True:
            try:

                start = time.time()

                # get next batch of data
                x_batch, y_batch = self.sess.run(next_batch)

                # Run the training op
                self.sess.run(train_op, feed_dict={self.x: x_batch,
                                                   self.y: y_batch})

                eta = (time.time() - start) * (nb - batch)
                eta = '%d:%02d' % (eta // 60, eta % 60)
                print("{}/{} ETA:{}".format(batch, nb, eta))
                batch += 1

            except tf.errors.OutOfRangeError:
                break


        self.sess.run(B_op)
        print("Training of ELM {} ended in {}:{:5f}".format(self.name, ((time.time() - t0) // 60),
                                                         ((time.time() - t0) % 60)))

        print("#"*100)

        train_metric = self.evaluate(dataset)

        if self.type is 'c':

            print('Train accuracy: ', train_metric)

        else: #regression

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
        self
        print("TensorFlow graph resetted")











