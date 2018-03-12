# Import dataset, from another files, splitted
import hpelm
import time
import numpy as np
from Misc.loadMNIST_orig import X_test, y_test, X_train, y_train
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


def AE_ELM(X, n_number, n_type='sigm', norm=None, B=None, W=None):
    AE = hpelm.ELM(X.shape[1], X.shape[1], batch=n_number, accelerator='GPU', precision='single', norm=norm)
    AE.add_neurons(n_number, n_type)
    print(str(AE))
    t = time.time()
    AE.train(X, X)
    time_train = time.time() - t
    y_train_predicted = AE.predict(X)
    print("Training time: %f" % time_train)
    print('Training Accuracy: ', (1 - AE.error(X, y_train_predicted)))
    B = AE.nnet.get_B()
    AE.nnet.reset()
    return B


def classificator_ELM(X_train, y_train, X_test, y_test, n_number, n_type='sigm', num_class=10):
    cls_elm = hpelm.ELM(X_train.shape[1], num_class, classification="c", batch=n_number, accelerator="GPU",
                        precision='single')
    cls_elm.add_neurons(n_number, n_type)
    print(str(cls_elm))
    # Training model
    t = time.time()
    cls_elm.train(X_train, y_train, 'c')
    elapsed_time_train = time.time() - t
    y_train_predicted = cls_elm.predict(X_train)
    print("Training time: %f" % elapsed_time_train)
    print('Training Accuracy: ', (1 - cls_elm.error(y_train, y_train_predicted)))
    # Prediction from trained model
    y_test_predicted = cls_elm.predict(X_test)
    print('Test Accuracy: ', (1 - cls_elm.error(y_test, y_test_predicted)))
    cls_elm.nnet.reset()


def activation(X, type):
    if type == 'sigm':
        H = np.exp(-np.logaddexp(0, -X))
    elif type == 'tanh':
        H = np.tanh(X)
    elif type == 'lin':
        H = X
    return H


###############################################################################################
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape and apply OneHotEncoder to compute the 10class classifier
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
onehot_encoder = OneHotEncoder(sparse=False)
y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)

# Hyperparameters
np.set_printoptions(precision=2)
np.random.seed(42)
neuron_number = 8192
out_class = 10
batch_size = neuron_number
'''
print("###############################################################################################")
print('ELM Normal, %d' % i)
elm = hpelm.ELM(X_train.shape[1], out_class, classification="c", batch=batch_size, accelerator="GPU",
                precision='single')
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
elm.nnet.reset()
del elm, y_train_predicted, y_test_predicted
print("###############################################################################################")
'''

print('\nELM-AE: Feature extraction')
num_layer = 2
AE_neurons = 700
for i in range(num_layer):
    if i == 0:
        B_AE = AE_ELM(X_train, AE_neurons, 'sigm', norm=10 ** -1)
        B_AE = np.transpose(B_AE)

        H_train = np.dot(X_train, B_AE)
        H_train = activation(H_train, 'sigm')

        H_test = np.dot(X_test, B_AE)
        H_test = activation(H_test, 'sigm')
    else:
        B_AE = AE_ELM(H_train, AE_neurons, 'sigm', norm=10 ** -3)
        B_AE = np.transpose(B_AE)

        H_train = np.dot(H_train, B_AE)
        H_train = activation(H_train, 'lin')

        H_test = np.dot(H_test, B_AE)
        H_test = activation(H_test, 'lin')

# Classificator output
print('\nELM Multiclass classificator')
neuron_number = 4096
ML_elm = hpelm.ELM(H_train.shape[1], out_class, classification="c", batch=neuron_number, accelerator="GPU",
                   precision='single', norm=10 ** -8)
ML_elm.add_neurons(neuron_number, 'sigm')
print(str(ML_elm))
# Training model
t = time.time()
ML_elm.train(H_train, y_train, 'c')
elapsed_time_train = time.time() - t
y_train_predicted = ML_elm.predict(H_train)
print("Training time: %f" % elapsed_time_train)
print('Training Accuracy: ', (1 - ML_elm.error(y_train, y_train_predicted)))
# Prediction from trained model
y_test_predicted = ML_elm.predict(H_test)
print('Test Accuracy: ', (1 - ML_elm.error(y_test, y_test_predicted)))
