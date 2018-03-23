'''Wrapper object to allow offline, non-real time image pre-processing
with keras ImageDataGenerator'''


import pickle
import time
import numpy as np


class keras_preprocessor(object):

    def __init__(self,
                 generator,
                 batch_size,

                 ):
        super(keras_preprocessor, self).__init__()

        self.generator = generator
        self.batch_size = batch_size




    def transform(self, instances, labels, inst_filepath, lab_filepath):

        num_batches =  instances.shape[0] // self.batch_size

        t = time.time()
        t0 = time.time()
        batches = 0

        for X_batch, y_batch in self.generator.flow(instances, labels,
                                         batch_size=self.batch_size):


            pickle.dump(X_batch, open(inst_filepath, 'ab'))
            pickle.dump(y_batch, open(lab_filepath, 'ab'))

            # report time
            eta = int(((time.time() - t0) / (batches + 1)) * (num_batches - batches - 1))
            if time.time() - t > 10:
                print('processing image batch %d/%d, eta: %d:%02d:%02d' % (
                    batches, num_batches, eta / 3600, (eta % 3600) / 60, eta % 60))
                t = time.time()

            batches += 1
            if batches >= num_batches:
                break





    def fit_transform(self, instances, labels, inst_filepath, lab_filepath):

        print("Fitting generator %s to data...." % self.generator)
        self.generator.fit(instances) # fitting to data
        self.transform(instances,labels,inst_filepath, lab_filepath)



    def load(self, inst_shape, lab_shape,  inst_filepath, lab_filepath):

        x_train_processed = np.empty((0, *inst_shape.shape[1:]))
        y_train_processed = np.empty((0, *lab_shape.shape[1:]))

        with open(inst_filepath, 'rb') as handle:
            while 1:
                try:
                    x_train_processed = np.append(x_train_processed, pickle.load(handle), axis=0)
                except EOFError:
                    break

        with open(lab_filepath, 'rb') as handle:
            while 1:
                try:
                    y_train_processed = np.append(y_train_processed, pickle.load(handle), axis=0)
                except EOFError:
                    break
        return x_train_processed,y_train_processed






def main(): # testing purpose only

    import os, sys, inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)
    from keras.datasets import cifar10

    import numpy as np

    from keras.preprocessing.image import ImageDataGenerator
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix

    (images_train, y_train), (images_test, y_test) = cifar10.load_data()

    os.makedirs(parentdir + '/cifar10_data' + '/augmented_data', exist_ok=True)
    save_dir = parentdir + '/cifar10_data' + '/augmented_data'
    x_train_proc = save_dir + '/x_train_proc.pickle'
    y_train_proc = save_dir + '/y_train_proc.pickle'

    x_test_proc = save_dir + '/x_test_proc.pickle'
    y_test_proc = save_dir + '/y_test_proc.pickle'

    gen = ImageDataGenerator(
        #samplewise_std_normalization=True,
        #samplewise_center=True,
        #featurewise_center=False,
        #featurewise_std_normalization=False,
        #zca_whitening=True,
        #zca_epsilon=1e-2,
        #shear_range=0.2,
        #channel_shift_range=0.2,
        #fill_mode='nearest',
        #rotation_range=2,
        #width_shift_range=0.2,
        #height_shift_range=0.2,
        #horizontal_flip=True,
        #vertical_flip=False
    )

    proc = keras_preprocessor(
        batch_size=200,
        generator=gen

    )


    proc.fit_transform(images_train,y_train,x_train_proc,y_train_proc)
    proc.transform(images_test,y_test,x_test_proc,y_test_proc)



    # load augmented dataset
    x_train_processed = np.empty((0, *images_train.shape[1:]))
    y_train_processed = np.empty((0, *y_train.shape[1:]))

    with open(x_train_proc, 'rb') as handle:
        while 1:
            try:
                x_train_processed = np.append(x_train_processed,pickle.load(handle), axis=0)
            except EOFError:
                break



    with open(y_train_proc, 'rb') as handle:
        while 1:
            try:
                y_train_processed = np.append(y_train_processed,pickle.load(handle), axis=0)
            except EOFError:
                break

    x_test_processed = np.empty((0, *images_train.shape[1:]))
    y_test_processed = np.empty((0, *y_train.shape[1:]))

    with open(x_test_proc, 'rb') as handle:
        while 1:
            try:
                x_test_processed = np.append(x_test_processed, pickle.load(handle), axis=0)
            except EOFError:
                break

    with open(y_test_proc, 'rb') as handle:
        while 1:
            try:
                y_test_processed = np.append(y_test_processed, pickle.load(handle), axis=0)
            except EOFError:
                break


    from images_utils import plot_images


    plot_images(images=x_train_processed[0:9],
                cls_true=y_train_processed[0:9])

    plot_images(images=x_test_processed[0:9],
                cls_true=y_test_processed[0:9])


    import hpelm
    from sklearn.preprocessing import OneHotEncoder
    onehot_encoder = OneHotEncoder(sparse=False)
    Y_train = onehot_encoder.fit_transform(y_train_processed)
    y_test = onehot_encoder.fit_transform(y_test_processed)
    X_train = x_train_processed.reshape((-1,32*32*3))
    X_test = x_test_processed.reshape((-1, 32 * 32 * 3))


    print('\nELM simple')
    elm = hpelm.ELM(X_train.shape[1], 10, classification="c", accelerator="GPU", precision='single')
    elm.add_neurons(8196, 'sigm')
    print(str(elm))
    # Training model
    t = time.time()
    elm.train(X_train, Y_train, 'c')
    elapsed_time_train = time.time() - t
    y_train_predicted = elm.predict(X_train)
    print("Training time: %f" % elapsed_time_train)
    print('Training Accuracy: ', (1 - elm.error(Y_train, y_train_predicted)))
    # Prediction from trained model
    y_test_predicted = elm.predict(X_test)
    print('Test Accuracy: ', (1 - elm.error(y_test, y_test_predicted)))
    # print(elm.confusion(y_test,y_test_predicted)) #value as 4E+5
    y_test_sk = y_test.argmax(1)
    y_test_predicted_sk = y_test_predicted.argmax(1)
    class_report = classification_report(y_test_sk, y_test_predicted_sk)
    cnf_matrix = confusion_matrix(y_test_sk, y_test_predicted_sk)
    print("Confusion Matrix:\n", cnf_matrix)
    print("Classification report\n: ", class_report)



if __name__ == '__main__':
        main()





