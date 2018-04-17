import tensorflow as tf
from datetime import datetime
import os, sys
import time
import numpy as np
import h5py


class elm(object):

    def __init__(
            self,
            input_size,
            output_size,
            savedir = None,
            type='c',
            name="elm",
            l2norm=None,
    ):

        self.n_neurons = [input_size, output_size]
        self.savedir = savedir
        self.activation = []
        self.name = name
        self.type = type
        self.metric = None
        self.HH = None
        self.HT = None
        self.H = []
        self.Hw = []
        self.Hb = []
        self.B = None
        self.y_out = None

        if l2norm is None:
            l2norm = 50 * np.finfo(np.float32).eps
        self.l2norm = l2norm

        # start tensorflow session
        self.sess = tf.Session()

        # define graph inputs
        with tf.name_scope("input_" + self.name):
            self.x = tf.placeholder(dtype='float32', shape=[None, input_size], name='x')
            self.y = tf.placeholder(dtype='float32', shape=[None, output_size], name='y')

        if self.savedir is not None:
            self.writer = tf.summary.FileWriter(self.savedir + "/" + self.name)

    def compile(self, n_neurons, activation=tf.sigmoid, w_init='default', b_init='default'):
        self.n_neurons.insert(-1, n_neurons)
        with tf.name_scope("hidden_layer_" + self.name):
            # TODO: better gestion of init_w and init_b
            if w_init is 'default' or b_init is 'default':
                init_w = tf.random_normal(shape=[self.n_neurons[0], self.n_neurons[1]],
                                          stddev=3. * tf.sqrt(tf.div(1., self.n_neurons[0])))

                init_b = tf.random_normal(shape=[self.n_neurons[1]],
                                          stddev=3. * tf.sqrt(tf.div(1., self.n_neurons[0])))

                self.Hb.append(tf.Variable(init_b, dtype=tf.float32, name='Hb', trainable=False))
                self.Hw.append(tf.Variable(init_w, dtype=tf.float32, name='Hw', trainable=False))

            else:
                # User must provide tensor object of Hb and Hw with tf.get_variable
                print("Using custom initialization for ELM: {}".format(self.name))
                assert w_init or b_init is 'default', "Both w_initializer and b_initializer " \
                                                      "should be provided when using custom initialization"
                with tf.name_scope("custom_initialization"):
                    assert sorted(w_init.shape.as_list()) == sorted([self.n_neurons[0],
                                                                             self.n_neurons[1]]),\
                        "Invalid shape for hidden layer weights tensor"
                    self.Hw.append(w_init)

                    assert b_init.shape.as_list()[0] == self.n_neurons[1],\
                        "Invalid shape for hidden layer biases tensor"
                    self.Hb.append(b_init)

            self.H.append(activation(tf.matmul(self.x, self.Hw[-1]) + self.Hb[-1]))

        with tf.name_scope('output_layer_' + self.name):
            self.B = tf.Variable(tf.zeros(shape=[self.n_neurons[1], self.n_neurons[-1]]), dtype='float32')
            self.y_out = tf.matmul(self.H[-1], self.B)

        # initialization
        self.sess.run([self.Hw[-1].initializer, self.Hb[-1].initializer])
        print("Network parameters have been initialized")

        if self.savedir is not None:
            self.writer.add_graph(self.sess.graph)

    def get_iterator(self, x, y, batch_size=1000):
        print('Creating dataset iterator')
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        # creating tensorflow iterator
        batched_dataset = dataset.batch(batch_size=batch_size)
        iterator = batched_dataset.make_initializable_iterator()
        self.sess.run(iterator.initializer)
        return iterator

    def evaluate(self, x, y, batch_size=1000, iterator=None): #TODO predict and evaluate can be combined actually
        if iterator is not None: # reset iterator
            self.sess.run(iterator.initializer)
        else:  # create iterator
            iterator = self.get_iterator(x, y, batch_size=batch_size)

        next_batch = iterator.get_next()
        metric_vect = []
        while True:
            try:
                x_batch, y_batch = self.sess.run(next_batch)
                #x_batch = (x_batch - np.mean(x_batch)) / np.std(x_batch)
                metric_vect.append(self.sess.run(self.metric, feed_dict={self.x: x_batch, self.y: y_batch}))
            except tf.errors.OutOfRangeError:
                break
        return np.mean(metric_vect, dtype=np.float64)

    # TODO: fix conflict between iterator and x y data, problems will rise with hdf5 dataset
    # maybe do a parse argument and pass num batch and size?
    def train(self, x, y, iterator=None, accuracy=True, batch_size=1000):
        # define training structure
        with tf.name_scope("training_" + self.name):
            # initialization and training graph definition
            self.HH = tf.Variable(tf.multiply(tf.eye(self.n_neurons[1], dtype=tf.float32),
                                              tf.cast(self.l2norm, tf.float32)), name='HH')

            self.HT = tf.Variable(tf.zeros([self.n_neurons[1], self.n_neurons[-1]]), name='HT')

            train_op = tf.group(
             tf.assign_add(self.HH, tf.matmul(self.H[-1], self.H[-1], transpose_a=True)),
             tf.assign_add(self.HT, tf.matmul(self.H[-1], self.y, transpose_a=True))
            )

            B_op = tf.assign(self.B, tf.matmul(tf.matrix_inverse(self.HH), self.HT))

        if self.type is 'c':
            # no need for cost function # TODO this can use the predict method actually
            with tf.name_scope("accuracy_" + self.name):
                correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_out, 1))
                self.metric = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        elif self.type is 'r': # regression
            with tf.name_scope("mean_squared_error_" + self.name):
                # nb possible numerical instability TODO: fix mse
                # https://stackoverflow.com/questions/41338509/tensorflow-mean-squared-error-loss-function
                self.metric = tf.reduce_mean(tf.squared_difference(self.y_out,self.y, name='mse'))
        else:
            raise ValueError("Invalid argument for type")

        if self.savedir is not None:
            # add to graph
            self.writer.add_graph(self.sess.graph)
            # Initialize a saver to store model checkpoints # TODO: saver
            #saver = tf.train.Saver()
            print("{} Open tensorboard at --logdir={}".format(datetime.now(), self.savedir))

        if iterator is not None: # reset iterator
            self.sess.run(iterator.initializer)
        else:  # create iterator
            iterator = self.get_iterator(x, y, batch_size=batch_size)

        next_batch = iterator.get_next()
        nb = int(np.ceil(x.shape[0]/batch_size))

        # initialize variables
        self.sess.run([self.HH.initializer, self.HT.initializer])

        t0 = time.time()
        print("{} Start training...".format(datetime.now()))

        batch = 1
        while True:
            try:
                start = time.time()
                # get next batch of data
                x_batch, y_batch = self.sess.run(next_batch)
                #x_batch = (x_batch - np.mean(x_batch)) / np.std(x_batch)
                # Run the training op
                self.sess.run(train_op, feed_dict={self.x: x_batch, self.y: y_batch})
                if batch % 25 == 0:
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

        if accuracy is True:
            train_metric = self.evaluate(x, y, iterator=iterator)
            if self.type is 'c':
                print('Train accuracy: ', train_metric)
            else:  #regression
                print('Train MSE: ', train_metric)
            return train_metric

    # TODO: rewrite the predict method
    def iter_predict(self, x, y, iterator=None, dataname=None, batch_size = 1000, **kwargs):
        if iterator is not None: # reset iterator
            self.sess.run(iterator.initializer)
        else:  # create iterator
            iterator = self.get_iterator(x, y, batch_size=batch_size)
        next_batch = iterator.get_next()

        # TODO: better hdf5 file manager
        if dataname is not None:  # write to hdf5 file
            path = kwargs.get('filepath', "%s" % (self.savedir + "/" + dataname + ".hdf5"))
            assert os.path.exists(path) is not True, "Error: The file to which predicted values " \
                                                     "will be appended already exist."
            with h5py.File(path, "a") as f:
                pred_dset = f.create_dataset(dataname,(0, *y.shape[1:]), maxshape=(None, *y.shape[1:]),
                                             dtype='float32', chunks=(batch_size,*y.shape[1:]))
                while True:
                    try:
                        x_batch, y_batch = self.sess.run(next_batch)
                        pred_dset.resize(pred_dset.shape[0] + batch_size, axis=0)
                        # append at the end
                        pred_dset[-batch_size:] = self.sess.run([self.y_out],
                                                                feed_dict={self.x: x_batch, self.y: y_batch})
                    except tf.errors.OutOfRangeError:
                        break
                #return pred_dset #TODO return ?
        else:
            y_out = []
            while True:
                try:
                    x_batch, y_batch = self.sess.run(next_batch)
                    y_out.append(self.sess.run(self.y_out, feed_dict={self.x: x_batch, self.y: y_batch}))
                except tf.errors.OutOfRangeError:
                    break

            return np.reshape(np.asarray(y_out), (y.shape))

    def get_Hw_Hb(self, layer_number = -1):
        Hw = self.Hw[layer_number].eval(session=self.sess)  # get hidden layer weights matrix
        if self.Hb[layer_number] is not None:
            Hb = self.Hb[layer_number].eval(session=self.sess)  # get hidden layer biases
        else:
            Hb = None
        return Hw, Hb

    def get_B(self):
        return self.B.eval(session=self.sess)

    def get_HH(self):
        return self.HH.eval(session=self.sess)

    def __del__(self):
        self.sess.close()
        #tf.reset_default_graph()
        print("TensorFlow graph resetted")











