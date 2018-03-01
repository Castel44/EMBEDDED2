import time

import hpelm
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from CIFAR10.cifar10dataset import train_data, train_labels, test_data, test_labels

X_train = train_data.astype('float32')
y_train = train_labels.astype('float32')
X_test = test_data.astype('float32')
y_test = test_labels.astype('float32')

# Convert data in greyscale
# X_train = rgb2gray(X_train)
# X_test = rgb2gray(X_test)

print('CIFAR 10 DATASET')
print('X_train shape ', X_train.shape)
print('y_train shape ', y_train.shape)
print('X_test shape ', X_test.shape)
print('y_test shape ', y_test.shape)
out_class = len(np.unique(y_test))
print('Num Classes: ', out_class)

'''
# Add flipped train dataset
X_train_flip = X_train[:,:,:,::-1]
y_train_flip = y_train
X_train = np.concatenate((X_train,X_train_flip), axis=0)
y_train = np.concatenate((y_train, y_train_flip), axis=0)

rnd_idx = np.random.permutation(len(X_train))
X_train = X_train[rnd_idx]
y_train = y_train[rnd_idx]

print('CIFAR 10 DATASET agumented')
print('X_train shape ', X_train.shape)
print('y_train shape ', y_train.shape)
print('X_test shape ', X_test.shape)
print('y_test shape ', y_test.shape)
'''

print('Reshape and scaling data (mean 0, std 1)')
X_train = X_train.reshape(
    (len(X_train), X_train.shape[1] * X_train.shape[2] * X_train.shape[3]))  # X_train.shape[3] if coloured
X_test = X_test.reshape((len(X_test), X_test.shape[1] * X_test.shape[2] * X_test.shape[3]))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print('Reshape and apply OneHotEncoder')
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
onehot_encoder = OneHotEncoder(sparse=False)
y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)

'''
# Seed for random num gen, high dependence of accuracy from those numbers.
rnd_seed = 42
np.random.seed(42)
'''

np.set_printoptions(precision=2)
neuron_number = 4096
CV_folds = 10
prec = "single"

print('\nELM simple')
elm = hpelm.ELM(X_train.shape[1], out_class, classification="c", accelerator="GPU", batch=256, precision='single')
elm.add_neurons(neuron_number, 'sigm')
# elm.add_neurons(X_train.shape[1],'lin')
# elm.add_neurons(500,'lin')
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
