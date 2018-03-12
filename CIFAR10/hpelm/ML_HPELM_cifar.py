import time
import hpelm
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from CIFAR10.cifar10dataset import train_data, train_labels, test_data, test_labels
import Misc.CUDA_calc as gpu


def ae_elm(X, n_number, n_type='sigm', norm=None, orth_init=False, X_test=None):
    "Only max 784 neuron if wanna use the ortho init"

    AE = hpelm.ELM(X.shape[1], X.shape[1], batch=n_number, accelerator='GPU', precision='single', norm=norm)
    if orth_init:
        B, W = ortho_B_W(n_number, X.shape[1])
        AE.add_neurons(n_number, n_type, B=np.asarray(B, dtype=np.float32, order='C'),
                       W=np.asarray(W, dtype=np.float32, order='C'))
    else:
        AE.add_neurons(n_number, n_type)
    print(str(AE))
    t = time.time()
    AE.train(X, X)
    time_train = time.time() - t
    y_train_predicted = AE.predict(X)
    print("Training time: %f" % time_train)
    print('Training Accuracy: ', (1 - AE.error(X, y_train_predicted)))
    if X_test is not None:
        print("Test Accuracy: ", (1 - AE.error(X_test, AE.predict(X_test))))
    B = AE.nnet.get_B()
    AE.nnet.reset()
    return B


def activation(X, type):
    if type not in ['sigm', 'tanh', 'lin']: raise ValueError('unsupported combination of activations types')

    if type == 'sigm':
        H = np.exp(-np.logaddexp(0, -X))
    elif type == 'tanh':
        H = np.tanh(X)
    elif type == 'lin':
        H = X
    return H


def activation_gpu(X, B, type):
    if type not in ['sigm_stable', 'sigm', 'tanh', 'lin']: raise ValueError(
        'unsupported combination of activations types')

    if type == 'lin':
        H = gpu._dev_lin(X, B.T)
    elif type == 'sigm':
        H = gpu._dev_sigm(X, B.T)
    elif type == 'sigm_stable':
        H = gpu._dev_sigm_stable(X, B.T)
    elif type == 'tanh':
        H = gpu._dev_tanh(X, B.T)
    return H


def ortho_B_W(number, inputs):
    """ Generate random orthogonal weigth matrix (W) and bias vector (B)
    Check with:
    np.allclose(np.dot(B, B.T), 1)
    >> True
    np.allclose(np.dot(Vt, Vt.T), np.eye(Vt.shape[0]))
    >> True """
    B = np.random.randn(number)
    B /= np.linalg.norm(B)
    X = np.random.random((number, inputs))
    U, _, Vt = np.linalg.svd(X, full_matrices=False)
    Vt = Vt.T
    return B, Vt


#####################################################################################################################
print("Loading Dataset: CIFAR10")
X_train = train_data.astype('float32')
y_train = train_labels.astype('float32')
X_test = test_data.astype('float32')
y_test = test_labels.astype('float32')

print('CIFAR 10 DATASET')
print('X_train shape ', X_train.shape)
print('y_train shape ', y_train.shape)
print('X_test shape ', X_test.shape)
print('y_test shape ', y_test.shape)
out_class = len(np.unique(y_test))
print('Num Classes: ', out_class)

print('Reshape and scaling data (mean 0, std 1)')
X_train = X_train.reshape(
    (len(X_train), X_train.shape[1] * X_train.shape[2] * X_train.shape[3]))  # X_train.shape[3] if coloured
X_test = X_test.reshape((len(X_test), X_test.shape[1] * X_test.shape[2] * X_test.shape[3]))

prescaler = StandardScaler()
X_train = prescaler.fit_transform(X_train)
X_test = prescaler.transform(X_test)

print('Reshape and apply OneHotEncoder')
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
onehot_encoder = OneHotEncoder(sparse=False)
y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)
#####################################################################################################################

print('\nELM-AE GPU')
num_layer = 2
AE_neurons = 2000
for i in range(num_layer):
    if i == 0:
        B_AE = ae_elm(X_train, AE_neurons, 'sigm', norm=10 ** 1, orth_init=True, X_test=X_test)
        H_train = activation_gpu(X_train, B_AE, 'sigm')
        H_test = activation_gpu(X_test, B_AE, 'sigm')
    else:
        B_AE = ae_elm(H_train, AE_neurons, 'sigm', norm=10 ** -3, orth_init=True, X_test=H_test)
        H_train = activation_gpu(H_train, B_AE, 'lin')
        H_test = activation_gpu(H_test, B_AE, 'lin')

postscaler = StandardScaler()
H_train = postscaler.fit_transform(H_train)
H_test = postscaler.transform(H_test)
#####################################################################################################################

print("\nBuilding hdf5 files")
# Convert data in HDF5 files
PATH = "C:\\Users\\Andrea\\PycharmProjects\\EMBEDDED\\CIFAR10\\cifar10_data\\"
hpelm.make_hdf5(H_train, PATH + "H_train.h5")
hpelm.make_hdf5(y_train, PATH + "y_train.h5")
hpelm.make_hdf5(H_test, PATH + "H_test.h5")
hpelm.make_hdf5(y_test, PATH + "y_test.h5")

np.set_printoptions(precision=2)
neuron_number = 15000
out_class = 10

# Classificator
print('\nBuilding model: HPELM with ')
model = hpelm.HPELM(H_train.shape[1], out_class, classification="c", batch=2048, accelerator="GPU", precision='single',
                    tprint=3, norm=10 ** -8)
model.add_neurons(neuron_number, 'sigm')
print(str(model))
t = time.time()
model.train(PATH + 'H_train.h5', PATH + 'y_train.h5', 'c')
elapsed_time_train = time.time() - t
print("Training time: %f" % elapsed_time_train)
model.predict(PATH + 'H_train.h5', PATH + 'y_train_pred.h5')
print('Training Accuracy: ', (1 - model.error(PATH + 'y_train.h5', PATH + 'y_train_pred.h5')))
model.predict(PATH + 'H_test.h5', PATH + 'y_test_pred.h5')
print('Test Accuracy: ', (1 - model.error(PATH + 'y_test.h5', PATH + 'y_test_pred.h5')))
