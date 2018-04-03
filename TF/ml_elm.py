from elm import elm
import tensorflow as tf
import numpy as np
import gc

class ml_elm(object):

    def __init__(self, input_size,
            output_size,
            savedir = None,
            type = 'c',
            name = "ml_elm" ):

        #super(ml_elm, self).__init__()
        self.name = name
        self.type = type
        self.n_neurons = [input_size,output_size]
        self.activation = []
        self.w_initializer = []
        self.b_initializer = []
        self.l2norm = []
        self.Hw = []
        self.Hb = []
        self.H = []
        self.n_layers = 0
        self.savedir = savedir

        # start tensorflow session
        self.sess = tf.Session()

        # define graph inputs
        with tf.name_scope("input_" + self.name):
            self.x = tf.placeholder(dtype='float32', shape=[None, input_size])
            self.y = tf.placeholder(dtype='float32', shape=[None, output_size])

        if self.savedir is not None:
            self.writer = tf.summary.FileWriter(self.savedir + "/" + self.name)





    def add_layer(self,n_neurons, activation=tf.sigmoid, w_init='default', b_init='default',l2norm = None):

        if l2norm is None:
            l2norm = 50 * np.finfo(np.float32).eps

        self.n_neurons.insert(-1, n_neurons)
        self.activation.append(activation)
        self.w_initializer.append(w_init)
        self.b_initializer.append(b_init)
        self.l2norm.append(l2norm)
        self.n_layers += 1


    def _compile_ae(self, layer_num): # private

        with tf.name_scope("layer_%d" % (layer_num)):

            if layer_num is 0 :

               if self.Hb[layer_num] is not None:
                   self.H.append(self.activation[layer_num](tf.matmul(self.x, self.Hw[layer_num]) + self.Hb[layer_num]))
               else:
                   self.H.append(self.activation[layer_num](tf.matmul(self.x, self.Hw[layer_num])))

            else:

                if self.Hb[layer_num] is not None:
                    self.H.append(
                        self.activation[layer_num](tf.matmul(self.H[layer_num-1], self.Hw[layer_num]) + self.Hb[layer_num]))
                else:
                    self.H.append(self.activation[layer_num](tf.matmul(self.H[layer_num-1], self.Hw[layer_num])))


            # initialization
            #if self.Hb[layer_num] is not None:
             #       self.sess.run([self.Hw[layer_num].initializer, self.Hb[layer_num].initializer])

            #else:
             #   self.sess.run([self.Hw[layer_num].initializer])



    def train(self, x, y, batch_size=1000):

        assert self.n_layers > 1, "The network should contain at least two layers"

        prev_layer_act = None

        with tf.name_scope("ae_training_" + self.name):

            for ae in range(self.n_layers-1):

                print("Training Autoencoder n:%d out of %d" % (ae+1, (self.n_layers-1)))

                auto_enc = elm(self.n_neurons[ae], self.n_neurons[ae],savedir = (self.savedir + ("/ae:%d" % (ae+1))),
                               type = 'r',
                               name = ("ae_num_%d" % (ae)),
                               l2norm=self.l2norm[ae])

                auto_enc.add_layer(self.n_neurons[ae+1], activation=self.activation[ae], w_init=self.w_initializer[ae],
                                   b_init=self.b_initializer[ae])

                auto_enc.compile()

                if prev_layer_act is None:

                   auto_enc.train(x, x, batch_size=batch_size)

                   self.Hw.append(tf.transpose(auto_enc.get_B()))

                   _, b = auto_enc.get_Hw_Hb()

                   self.Hb.append(b)

                   del auto_enc

                   gc.collect()

                   self._compile_ae(ae)

                   prev_layer_act = self.sess.run(self.H[ae], feed_dict={self.x:x})

                else:

                    auto_enc.train(prev_layer_act, prev_layer_act, batch_size=batch_size)

                    self.Hw.append(tf.transpose(auto_enc.get_B()))


                    _, b = auto_enc.get_Hw_Hb()

                    self.Hb.append(b)

                    del auto_enc

                    gc.collect()

                    self._compile_ae(ae)

                    prev_layer_act = self.sess.run(self.H[ae], feed_dict={self.x:x}) # TODO use assign

        #tf.reset_default_graph()


        del self.H
        del prev_layer_act
        gc.collect()
        #with tf.name_scope("ml_elm_training_" + self.name):

        multi_layer_elm = elm(self.n_neurons[0], self.n_neurons[-1], savedir = (self.savedir + "/" + self.name ),
                               type = self.type,
                               name = self.name,
                               l2norm=self.l2norm[-1])


        for ae in range(self.n_layers-1):

            if self.Hb[ae] is not None:
                multi_layer_elm.add_layer(self.n_neurons[ae+1], activation=self.activation[ae],
                                        w_init=tf.Variable(self.Hw[ae]),
                                        b_init=tf.Variable(self.Hb[ae]))

            else:
                multi_layer_elm.add_layer(self.n_neurons[ae + 1], activation=self.activation[ae],
                                          w_init=tf.Variable(self.Hw[ae]),
                                          b_init=None)

        multi_layer_elm.add_layer(self.n_neurons[-2], activation=self.activation[-1],
                                  w_init=self.w_initializer[-1],
                                  b_init=self.b_initializer[-1])

        multi_layer_elm.compile()

        del self.Hb,self.Hw,self.activation,self.l2norm
        gc.collect()

        multi_layer_elm.train(x, y, batch_size=batch_size)

        return multi_layer_elm


def main():

    import keras
    import os
    from sklearn.preprocessing import StandardScaler

    mnist = keras.datasets.mnist
    train, test = mnist.load_data()
    x_train, y_train = train
    x_test, y_test = test
    del train, test
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)
    x_train = x_train.reshape(-1, 28 * 28)
    x_test = x_test.reshape(-1, 28 * 28)

    prescaler = StandardScaler()
    x_train = prescaler.fit_transform(x_train)
    x_test = prescaler.transform(x_test)



    ml_elm1 = ml_elm(784,10, savedir=(os.getcwd()+ "/ml_elm"))
    #ml_elm1.add_layer(700, l2norm=None, b_init=None)
    ml_elm1.add_layer(700, l2norm=None, b_init=None)
    ml_elm1.add_layer(1000)
    ml_elm1 =  ml_elm1.train(x_train, y_train, batch_size=500)
    acc = ml_elm1.evaluate(x_test,y_test)
    print(acc)
    
    del ml_elm1



    elm1 = elm(784,10)
    elm1.add_layer(1000)
    elm1.compile()
    elm1.train(x_train,y_train)
    acc =  elm1.evaluate(x_test,y_test)
    print(acc)


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