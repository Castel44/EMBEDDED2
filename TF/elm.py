import tensorflow as tf
from datetime import datetime
import os
import time
import numpy as np
import h5py


class elm(object):

    def __init__(
            self,
            input_size,
            output_size,
            savedir = None,
            type = 'c',
            name = "elm",
            l2norm=None, #TODO with tf precision

    ):

        self.n_neurons = [input_size,output_size]
        self.w_initializer = []
        self.b_initializer = []
        self.savedir = savedir
        self.activation = []
        self.name = name
        self.type = type
        self.metric = None
        self.HH = None
        self.HT = None
        self.n_hidden_layer = 0
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
            self.x = tf.placeholder(dtype='float32', shape=[None, input_size])
            self.y = tf.placeholder(dtype='float32', shape=[None, output_size])

        if self.savedir is not None:
            self.writer = tf.summary.FileWriter(self.savedir + "/" + self.name)

    def add_layer(self, n_neurons, activation=tf.sigmoid, w_init='default', b_init='default'):
        # add an hidden layer
        self.n_neurons.insert(-1, n_neurons)
        self.activation.append(activation)
        self.w_initializer.append(w_init)
        self.b_initializer.append(b_init)
        self.n_hidden_layer += 1

    def compile(self):
        assert self.n_hidden_layer is not 0, "Before compiling the network at least one hidden layer should be created"
        for layer in range(self.n_hidden_layer):

            with tf.name_scope("hidden_layer_" + self.name + ("_%d" % layer)):
                if self.w_initializer[layer] is 'default' or self.b_initializer[layer] is 'default':
                    init_w = tf.random_normal(shape=[self.n_neurons[layer], self.n_neurons[layer+1]],
                                              stddev=tf.sqrt(tf.div(2.,
                                                                    tf.add(tf.cast(self.n_neurons[layer-1], 'float32'),
                                                                           tf.cast(self.n_neurons[layer+2], 'float32')))))

                    if self.b_initializer[layer] is not None:
                        init_b = tf.random_normal(shape=[self.n_neurons[layer + 1]],
                                                  stddev=tf.sqrt(tf.div(2.,
                                                                        tf.add(tf.cast(self.n_neurons[layer - 1],
                                                                                       'float32'),
                                                                               tf.cast(self.n_neurons[layer + 2],
                                                                                       'float32')))))

                        self.Hb.append(tf.Variable(init_b))
                    else:
                        self.Hb.append(None)



                    self.Hw.append(tf.Variable(init_w))

                else:
                    print("Using custom inizialization for ELM: {} and layer number {}/{}".format(self.name, layer+1, self.n_hidden_layer))

                    with tf.name_scope("custom_initialization_" + ("_%d" % layer)):
                        self.Hw.append(self.w_initializer[layer])
                        assert self.Hw[layer] or self.Hb[layer] is 'default', "Both w_initializer and b_initializer " \
                            "should be provided when using custom initialization"
                        assert sorted(self.Hw[layer].shape.as_list()) == sorted([self.n_neurons[layer],
                                                                                 self.n_neurons[layer + 1]]),\
                            "Invalid shape for hidden layer weights tensor"

                        if self.b_initializer[layer] is not None:  # check
                            self.Hb.append(self.b_initializer[layer])
                            assert self.Hb[layer].shape.as_list()[0] == self.n_neurons[layer + 1],\
                                "Invalid shape for hidden layer biases tensor"
                        else:
                            self.Hb.append(None)

                if layer == 0:
                    if self.Hb[layer] is not None:
                        self.H.append(self.activation[layer](tf.matmul(self.x, self.Hw[layer]) + self.Hb[layer]))
                    else:
                        self.H.append(self.activation[layer](tf.matmul(self.x, self.Hw[layer])))
                else:
                    if self.Hb[layer] is not None:
                        self.H.append(self.activation[layer](tf.matmul(self.H[layer-1], self.Hw[layer]) + self.Hb[layer]))
                    else:
                        self.H.append(self.activation[layer](tf.matmul(self.H[layer - 1], self.Hw[layer])))

                # initialization
                if self.Hb[layer] is not None:
                    self.sess.run([self.Hw[layer].initializer, self.Hb[layer].initializer])

                else:
                    self.sess.run([self.Hw[layer].initializer])

        with tf.name_scope('output_layer_' + self.name):
            self.B = tf.Variable(tf.zeros(shape=[self.n_neurons[self.n_hidden_layer], self.n_neurons[-1]]),
                             dtype='float32')
            self.y_out = tf.matmul(self.H[self.n_hidden_layer - 1], self.B)

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

    def evaluate(self, x, y, batch_size =1000, iterator=None): #TODO predict and evaluate can be combined actually
        if iterator is not None: # reset iterator
            self.sess.run(iterator.initializer)
        else:  # create iterator
            iterator = self.get_iterator(x, y, batch_size=batch_size)

        next_batch = iterator.get_next()

        metric_vect = []

        while True:
            try:
                x_batch, y_batch = self.sess.run(next_batch)
                metric_vect.append(self.sess.run(self.metric,feed_dict={self.x: x_batch, self.y: y_batch}))
            except tf.errors.OutOfRangeError:
                break
        return np.mean(metric_vect)

    def train(self, x, y,  batch_size=1000):
        # define training structure
        with tf.name_scope("training_" + self.name):
            # initialization and training graph definition
            self.HH = tf.Variable(tf.multiply(tf.eye(self.n_neurons[self.n_hidden_layer], dtype=tf.float32),
                                              tf.cast(self.l2norm, tf.float32)),
                                              name='HH')

            self.HT = tf.Variable(tf.zeros([self.n_neurons[self.n_hidden_layer], self.n_neurons[-1]]), name='HT')

            train_op = tf.group(
             tf.assign_add(self.HH, tf.matmul(self.H[self.n_hidden_layer-1], self.H[self.n_hidden_layer-1], transpose_a=True)),
             tf.assign_add(self.HT, tf.matmul(self.H[self.n_hidden_layer-1], self.y, transpose_a=True))
            )

            B_op = tf.assign(self.B, tf.matmul(tf.matrix_inverse(self.HH), self.HT))

        if self.type is 'c':
            # no need for cost function # TODO this can use the predict method actually
            with tf.name_scope("accuracy_" + self.name):
                correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_out, 1))
                self.metric = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        elif self.type is 'r': # regression
            with tf.name_scope("mean_squared_error_" + self.name):
                # nb possible numerical instability
                # https://stackoverflow.com/questions/41338509/tensorflow-mean-squared-error-loss-function
                # TODO
                self.metric = tf.reduce_mean(tf.squared_difference(self.y_out,self.y, name='mse'))
        else:
            raise ValueError("Invalid argument for type")

        if self.savedir is not None:
            # add to graph
            self.writer.add_graph(self.sess.graph)
            # Initialize a saver to store model checkpoints # TODO saver
            saver = tf.train.Saver()
            print("{} Open tensorboard at --logdir={}".format(datetime.now(),
                                                              self.savedir))

        iterator = self.get_iterator(x, y, batch_size)
        next_batch = iterator.get_next()

        nb = int(np.ceil(x.shape[0]/batch_size))     #TODO

        # initialize variables
        self.sess.run([self.HH.initializer, self.HT.initializer])
        self.sess.run(iterator.initializer)

        t0 = time.time()
        print("{} Start training...".format(datetime.now()))

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

        train_metric = self.evaluate(x,y, iterator=iterator)
        if self.type is 'c':
            print('Train accuracy: ', train_metric)
        else:  #regression
            print('Train MSE: ', train_metric)
        return train_metric

    def iter_predict(self, x, y, dataname=None, batch_size = 10000, **kwargs):
        iterator = self.get_iterator(x, y, batch_size=batch_size)
        next_batch = iterator.get_next()

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

            return np.reshape(np.asarray(y_out), ((y.shape)))

    def get_Hw_Hb(self, layer_number = 0):
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











