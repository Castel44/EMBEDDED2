import pickle
import time

import hpelm
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


def unpickle(file):
    '''Load byte data from file'''
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='latin-1')
        return data


def load_cifar10_data(data_dir):
    '''Return train_data, train_labels, test_data, test_labels
    The shape of data is 32 x 32 x3'''
    train_data = None
    train_labels = []

    for i in range(1, 6):
        data_dic = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            train_data = data_dic['data']
        else:
            train_data = np.vstack((train_data, data_dic['data']))
        train_labels += data_dic['labels']

    test_data_dic = unpickle(data_dir + "/test_batch")
    test_data = test_data_dic['data']
    test_labels = test_data_dic['labels']
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    return train_data, train_labels, test_data, test_labels


data_dir = "F:\\Documenti 2\\University\\Magistrale\\Progettazione Sistemi Embedded\\Progetto EMBEDDED\\Datasets\\cifar-10-batches-py"

X_train, y_train, X_test, y_test = load_cifar10_data(data_dir)

print('CIFAR 10 DATASET')
print('X_train shape ', X_train.shape)
print('y_train shape ', y_train.shape)
print('X_test shape ', X_test.shape)
print('y_test shape ', y_test.shape)
out_class = len(np.unique(y_test))
print('Num Classes: ', out_class)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print('Reshape and apply OneHotEncoder')
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
onehot_encoder = OneHotEncoder(sparse=False)
y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)

# Seed for random num gen, high dependence of accuracy from those numbers.
# rnd_seed = 42
# np.random.seed(42)

np.set_printoptions(precision=2)
neuron_number = 1024
CV_folds = 10

print('\nELM simple')
elm = hpelm.ELM(X_train.shape[1], out_class, classification="c", accelerator="GPU", precision='single')
elm.add_neurons(neuron_number, 'sigm')
print(str(elm))
# Training model
t = time.time()
elm.train(X_train, y_train, 'c')
elapsed_time_train = time.time() - t
y_train_predicted = elm.predict(X_train)
print("Training time: %f" % elapsed_time_train)
print('Training Accuracy: ', (1 - elm.error(y_train, y_train_predicted)))
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
