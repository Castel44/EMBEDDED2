from TF.elm import elm
import numpy as np
import tensorflow as tf
from datetime import datetime
import time


class ml_elm(elm):
    def __init__(self,input_size,
            output_size,
            savedir = None,
            type = 'c',
            name = "ml_elm",
            ):

        super(ml_elm, self).__init__(input_size,
            output_size,
            savedir = savedir,
            type = type,
            name = name,
            )

        self.l2norm=[]
        self.ae_B = [] # overrides base class
        self.ae_y_out = []
        self.ae_Hw = []
        self.ae_H = []

    def add_layer(self, n_neurons, activation=tf.tanh, w_init='default', b_init='default', l2norm=None):

        if l2norm is None:
            l2norm = 50 * np.finfo(np.float32).eps

        self.n_neurons.insert(-1, n_neurons)
        self.activation.append(activation)
        self.w_initializer.append(w_init)
        self.b_initializer.append(b_init)
        self.l2norm.append(l2norm)
        self.n_hidden_layer += 1

    def _compile_ae(self,layer):

        with tf.name_scope("autoenc_of_" + self.name + ("_n_%d" % layer)):
            if self.w_initializer[layer] is 'default' or self.b_initializer[layer] is 'default':
                init_w = tf.random_normal(shape=[self.n_neurons[layer], self.n_neurons[layer + 1]],
                                          stddev=3.*tf.sqrt(tf.div(1., self.n_neurons[layer-1])))
                                                                #tf.add(
                                                                 #   tf.cast(self.n_neurons[layer - 1], 'float32'),
                                                                  # tf.cast(self.n_neurons[layer + 1],
                                                                   #         'float32')))))

                if self.b_initializer[layer] is not None:
                    init_b = tf.random_normal(shape=[self.n_neurons[layer + 1]],
                                              stddev=3.*tf.sqrt(tf.div(1., self.n_neurons[layer-1])))
                                                                    #tf.add(tf.cast(self.n_neurons[layer - 1],
                                                                     #              'float32'),
                                                                      #     tf.cast(self.n_neurons[layer + 1],
                                                                       #            'float32')))))

                    self.Hb.append(tf.Variable(init_b))
                else:
                    self.Hb.append(None)

                self.ae_Hw.append(tf.Variable(init_w))

            else:
                print(
                    "Using custom inizialization for AE-ELM: {} and layer number {}/{}".format(self.name, layer + 1,
                                                                                               self.n_hidden_layer))

                with tf.name_scope("custom_initialization_" + ("_%d" % layer)):

                    self.ae_Hw.append(self.w_initializer[layer])

                    assert self.ae_Hw[layer] or self.Hb[layer] is 'default', "Both w_initializer and b_initializer " \
                                                                             "should be provided when using custom initialization"

                    assert sorted(self.ae_Hw[layer].shape.as_list()) == sorted([self.n_neurons[layer],
                                                                                self.n_neurons[layer + 1]]), \
                        "Invalid shape for hidden layer weights tensor"

                    if self.b_initializer[layer] is not None:  # check
                        self.Hb.append(self.b_initializer[layer])
                        assert self.Hb[layer].shape.as_list()[0] == self.n_neurons[layer + 1], \
                            "Invalid shape for hidden layer biases tensor"
                    else:
                        self.Hb.append(None)

            if layer == 0:
                if self.Hb[layer] is not None:
                    self.ae_H.append(self.activation[layer](tf.matmul(self.x, self.ae_Hw[layer]) + self.Hb[layer]))
                else:
                    self.ae_H.append(self.activation[layer](tf.matmul(self.x, self.ae_Hw[layer])))
            else:
                if self.Hb[layer] is not None:
                    self.ae_H.append(
                        self.activation[layer](tf.matmul(self.H[layer - 1], self.ae_Hw[layer]) + self.Hb[layer]))
                else:
                    self.ae_H.append(self.activation[layer](tf.matmul(self.H[layer - 1], self.ae_Hw[layer])))

                    # instead of input take as input the activation of the first layer of the final structure

            # initialization
            if self.Hb[layer] is not None:
                self.sess.run([self.ae_Hw[layer].initializer, self.Hb[layer].initializer])

            else:
                self.sess.run([self.ae_Hw[layer].initializer])

            with tf.name_scope('output_layer_' + self.name):
                self.ae_B.append(tf.Variable(tf.zeros(shape=[self.n_neurons[layer + 1], self.n_neurons[layer]]),
                                             dtype='float32'))

                self.ae_y_out.append(tf.matmul(self.ae_H[layer], self.ae_B[layer]))

            print("Network parameters have been initialized")


    def _compile_layer(self,layer):

        self.Hw.append(tf.transpose(self.ae_B[layer]))

        with tf.name_scope("hidden_layer_of_" + self.name + ("_n_%d" % layer)):
            if layer == 0:
                if self.Hb[layer] is not None:
                    self.H.append(self.activation[layer](tf.matmul(self.x, self.Hw[layer]) + self.Hb[layer]))
                else:
                    self.H.append(self.activation[layer](tf.matmul(self.x, self.Hw[layer])))
            else:
                if self.Hb[layer] is not None:
                    self.H.append(
                        self.activation[layer](tf.matmul(self.H[layer - 1], self.Hw[layer]) + self.Hb[layer]))
                else:
                    self.H.append(self.activation[layer](tf.matmul(self.H[layer - 1], self.Hw[layer])))

    def compile(self):
        pass

    def _train_ae(self,layer, iterator, nb):

        with tf.name_scope("ae_training_n_%d_of_%s" % (layer, self.name)):

            self.HH = tf.Variable(tf.multiply(tf.eye(self.n_neurons[layer + 1], dtype=tf.float32),
                                              tf.cast(self.l2norm[layer], tf.float32)),
                                  name='HH')

            self.HT = tf.Variable(tf.zeros([self.n_neurons[layer + 1], self.n_neurons[layer]]), name='HT')

            if layer == 0:

                train_op = tf.group(
                    tf.assign_add(self.HH, tf.matmul(self.ae_H[layer],
                                                     self.ae_H[layer],
                                                     transpose_a=True)),
                    tf.assign_add(self.HT, tf.matmul(self.ae_H[layer], self.x,
                                                     transpose_a=True))
                )

            else:

                train_op = tf.group(
                    tf.assign_add(self.HH, tf.matmul(self.ae_H[layer],
                                                     self.ae_H[layer],
                                                     transpose_a=True)),
                    tf.assign_add(self.HT, tf.matmul(self.ae_H[layer], self.H[layer - 1],
                                                     transpose_a=True))
                )

            B_op = tf.assign(self.ae_B[layer], tf.matmul(tf.matrix_inverse(self.HH), self.HT))

            with tf.name_scope("mean_squared_error"):
                # nb possible numerical instability
                # https://stackoverflow.com/questions/41338509/tensorflow-mean-squared-error-loss-function
                # TODO
                if layer == 0:
                    metric = tf.reduce_mean(tf.squared_difference(self.ae_y_out[layer], self.x, name='mse'))
                else:
                    metric = tf.reduce_mean(tf.squared_difference(self.ae_y_out[layer], self.H[layer-1], name='mse'))



            self.sess.run([self.HH.initializer, self.HT.initializer])

            next_batch = iterator.get_next()

            self.sess.run(iterator.initializer)

            t0 = time.time()
            print("{} Start training...".format(datetime.now()))

            batch = 1
            while True:
                try:
                    start = time.time()
                    # get next batch of data
                    x_batch, y_batch = self.sess.run(next_batch)

                    # Run the training op

                    self.sess.run(train_op, feed_dict={self.x: x_batch})

                    eta = (time.time() - start) * (nb - batch)
                    eta = '%d:%02d' % (eta // 60, eta % 60)
                    print("{}/{} ETA:{}".format(batch, nb, eta))
                    batch += 1
                except tf.errors.OutOfRangeError:
                    break

            self.sess.run(B_op)
            metric = self.sess.run(metric, feed_dict={self.x: x_batch})

        print("Training of AE {} ended in {}:{:5f}".format(self.name, ((time.time() - t0) // 60),
                                                            ((time.time() - t0) % 60)))

        print("MSE: %.7f" % metric)
        print("#" * 100)






    def train(self, x, y, batch_size=1000):

        assert self.n_hidden_layer > 1, "Before compiling the network at least two hidden layers should be created"

        iterator = self.get_iterator(x, x, batch_size)

        nb = int(np.ceil(x.shape[0] / batch_size))


        for layer in range(self.n_hidden_layer-1):
            self._compile_ae(layer)
            self._compile_layer(layer)
            self._train_ae(layer, iterator, nb)


        # initialize and compile last layer

        with tf.name_scope("ELM_layer_" + self.name + ("_%d" % self.n_hidden_layer)):
            if self.w_initializer[-1] is 'default' or self.b_initializer[-1] is 'default':
                init_w = tf.random_normal(shape=[self.n_neurons[-3], self.n_neurons[-2]],
                                          stddev=3.*tf.sqrt(tf.div(1., self.n_neurons[-3])))
                                                              # tf.add(tf.cast(self.n_neurons[-3], 'float32'),
                                                               #        tf.cast(self.n_neurons[-1], 'float32')))))

                if self.b_initializer[-1] is not None:
                    init_b = tf.random_normal(shape=[self.n_neurons[-2]],
                                              stddev=3.*tf.sqrt(tf.div(1., self.n_neurons[-3])))
                                                                  # tf.add(tf.cast(self.n_neurons[-3],
                                                                   #              'float32'),
                                                                    #      tf.cast(self.n_neurons[-1],
                                                                     #            'float32')))))

                    self.Hb.append(tf.Variable(init_b))
                else:
                    self.Hb.append(None)

                self.Hw.append(tf.Variable(init_w))

            else:
                print("Using custom inizialization for ELM: {} and layer number {}/{}".format(self.name,
                                                                                              self.n_hidden_layer,
                                                                                              self.n_hidden_layer))

                with tf.name_scope("custom_initialization_" + ("_%d" % self.n_hidden_layer)):
                    self.Hw.append(self.w_initializer[-1])
                    assert self.Hw[-1] or self.Hb[-1] is 'default', "Both w_initializer and b_initializer " \
                                                                    "should be provided when using custom initialization"
                    assert sorted(self.Hw[-1].shape.as_list()) == sorted([self.n_neurons[-3],
                                                                          self.n_neurons[-2]]), \
                        "Invalid shape for hidden layer weights tensor"

                    if self.b_initializer[-1] is not None:  # check
                        self.Hb.append(self.b_initializer[-1])
                        assert self.Hb[-1].shape.as_list()[0] == self.n_neurons[-2], \
                            "Invalid shape for hidden layer biases tensor"
                    else:
                        self.Hb.append(None)

            if self.Hb[-1] is not None:

                self.sess.run([self.Hw[-1].initializer, self.Hb[-1].initializer])

            else:
                self.sess.run([self.Hw[-1].initializer])


            if self.Hb[layer] is not None:
                    self.H.append(self.activation[layer](tf.matmul(self.H[-1], self.Hw[-1]) + self.Hb[-1]))
            else:
                    self.H.append(self.activation[layer](tf.matmul(self.H[-1], self.Hw[-1])))



        with tf.name_scope('output_layer_' + self.name):
            self.B = tf.Variable(tf.zeros(shape=[self.n_neurons[-2], self.n_neurons[-1]]),
                                 dtype='float32')
            self.y_out = tf.matmul(self.H[-1], self.B)

        iterator = self.get_iterator(x, y, batch_size)

        # last layer training

        # define training structure
        with tf.name_scope("training_" + self.name):
            # initialization and training graph definition
            self.HH = tf.Variable(tf.multiply(tf.eye(self.n_neurons[-2], dtype=tf.float32),
                                              tf.cast(self.l2norm[-1], tf.float32)),
                                  name='HH')

            self.HT = tf.Variable(tf.zeros([self.n_neurons[-2], self.n_neurons[-1]]), name='HT')

            train_op = tf.group(
                tf.assign_add(self.HH, tf.matmul(self.H[-1], self.H[-1],
                                                 transpose_a=True)),
                tf.assign_add(self.HT, tf.matmul(self.H[-1], self.y, transpose_a=True))
            )

            B_op = tf.assign(self.B, tf.matmul(tf.matrix_inverse(self.HH), self.HT))

        if self.type is 'c':
            # no need for cost function # TODO this can use the predict method actually
            with tf.name_scope("accuracy_" + self.name):
                correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_out, 1))
                self.metric = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        elif self.type is 'r':  # regression
            with tf.name_scope("mean_squared_error_" + self.name):
                # nb possible numerical instability
                # https://stackoverflow.com/questions/41338509/tensorflow-mean-squared-error-loss-function
                # TODO
                self.metric = tf.reduce_mean(tf.squared_difference(self.y_out, self.y, name='mse'))
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

        nb = int(np.ceil(x.shape[0] / batch_size))  # TODO

        # initialize variables
        self.sess.run([self.HH.initializer, self.HT.initializer])

        t0 = time.time()
        print("{} Start training...".format(datetime.now()))


        self.sess.run(iterator.initializer)

        batch = 1
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
        print("#" * 100)

        train_metric = self.evaluate(x, y, iterator=iterator)
        if self.type is 'c':
            print('Train accuracy: ', train_metric)
        else:  # regression
            print('Train MSE: ', train_metric)
        return train_metric


def main():
    import keras
    import os
    from sklearn.preprocessing import StandardScaler

    mnist = keras.datasets.mnist
    train, test = mnist.load_data(os.getcwd()+ "/elm_tf_test"+ "mnist.txt")
    x_train, y_train = train
    x_test, y_test = test
    del train, test
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)
    x_train = x_train.reshape(-1, 28 * 28)
    x_test = x_test.reshape(-1, 28 * 28)

    #prescaler = StandardScaler()
    #x_train = prescaler.fit_transform(x_train)
    #x_test = prescaler.transform(x_test)
    #del prescaler

    input_size = 784
    output_size = 10

    # https://www.wolframalpha.com/input/?i=plot+sign(x)*((k*abs(x)))%2F(k-a*abs(x)%2B1),+x%3D-255..255,++k%3D-100,+a%3D1
    def tunable_sigm(t):
        k = -100.
        a = 1.
        half = tf.div(tf.multiply(k,tf.abs(t)), tf.add(1.,tf.add(k, -a*tf.abs(t))))
        return tf.multiply(tf.sign(t), half)

    init_w = tf.Variable(tf.truncated_normal(stddev=0.5, shape=[input_size,700]))
    init_b = tf.Variable(tf.truncated_normal(stddev=0.5, shape=[700]))
    init_w2 = tf.Variable(tf.truncated_normal(stddev=0.5, shape=[700,700]))

    #def sigm_zero_mean(x):
     #   return tf.subtract(1., tf.sigmoid(x))

    #def dual_relu(x):

     #   return tf.sign(x)*tf.nn.relu(x)

    ml_elm1 = ml_elm(input_size, output_size, savedir=(os.getcwd() + "/ml_elm"))
    ml_elm1.add_layer(700, activation=tunable_sigm)
    ml_elm1.add_layer(700, activation=tunable_sigm)
    ml_elm1.add_layer(2000, activation=tunable_sigm)
    ml_elm1.train(x_train, y_train, batch_size=500)
    acc = ml_elm1.evaluate(x_test, y_test)

    print(acc)

    del ml_elm1
    '''

    elm1 = elm(784, 10)
    elm1.add_layer(4000)
    elm1.compile()
    elm1.train(x_train, y_train)
    acc = elm1.evaluate(x_test, y_test)
    print(acc)
    '''

    '''
    import hpelm
    elm = hpelm.ELM(x_train.shape[1], 784, classification="r", accelerator="GPU", precision='single', batch=500)
    elm.add_neurons(4000, 'sigm')
    print(str(elm))
    # Training model
    elm.train(x_train, x_train, 'r')
    y_train_predicted = elm.predict(x_train)
    print('Training Error: ', (elm.error(x_train, y_train_predicted)))

    '''


if __name__ == '__main__':

    main()





