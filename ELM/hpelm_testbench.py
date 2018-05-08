import hpelm
import numpy as np
import keras
import os
import time
import itertools


def load_mnist():
    from keras.datasets import mnist
    print('Loading MNIST dataset')
    train, test = mnist.load_data()
    x_train, y_train = train
    x_test, y_test = test
    del train, test
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)
    x_train = x_train.reshape(-1, 28 * 28).astype('float32')
    x_test = x_test.reshape(-1, 28 * 28).astype('float32')
    img_size = 28
    img_channels = 1
    return x_train, x_test, y_train, y_test, img_size, img_channels


def load_cifar():
    from keras.datasets import cifar10
    print("Loading Dataset: CIFAR10")
    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)
    x_train = x_train.reshape(-1, 32 * 32 * 3).astype('float32')
    x_test = x_test.reshape(-1, 32 * 32 * 3).astype('float32')
    img_size = 32
    img_channels = 3
    return x_train, x_test, y_train, y_test, img_size, img_channels


######################################################################################################################
def main():
    # Get dataset
    x_train, x_test, y_train, y_test, img_size, img_channels = load_mnist()

    # Data scaler
    from sklearn.preprocessing import StandardScaler
    prescaler = StandardScaler()
    x_train = prescaler.fit_transform(x_train)
    x_test = prescaler.transform(x_test)


    # Hyperparameters
    input_size = img_size**2 * img_channels
    output_size = 10
    n_neurons = (1000, 5000, 8000, 15000)
    batch_size = 1000
    n_epochs = 1
    norm = (None,)

    ######################################################################################################################

    train_time = []
    train_acc = []
    test_acc = []
    run = 0
    run_comb = list(itertools.product(n_neurons, norm))
    for v in itertools.product(n_neurons, norm):
        print('\nStarting run %d/%d' % (run + 1, run_comb.__len__()))
        print('Hyperpar: neurons= ', v[0], 'norm=', v[1])
        elm = hpelm.ELM(input_size, output_size, classification="c", batch=batch_size, accelerator="GPU",
                        precision='single')
        elm.add_neurons(v[0], 'sigm')
        print(str(elm))
        # Training model
        t = time.time()
        elm.train(x_train, y_train, 'c')
        train_time.append(time.time() - t)
        train_acc.append(1 - elm.error(y_train, elm.predict(x_train)))
        print("Training time: %f" % train_time[run])
        print('Training Accuracy: ', train_acc[run])
        # Prediction from trained model
        test_acc.append(1 - elm.error(y_test, elm.predict(x_test)))
        print('Test Accuracy: ', test_acc[run])
        run += 1

    print('\nDone training!')
    # Searching for best hypar combination
    best_net = np.argmax(test_acc)
    print('Best net with hypepar:')
    print('  -neuron number:', run_comb[best_net][0])
    print('  -norm:', run_comb[best_net][2])
    print('Best net test accuracy: ', test_acc[best_net])


if __name__ == '__main__':
    main()