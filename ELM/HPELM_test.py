import numpy as np
from sklearn.datasets import fetch_mldata
import hpelm
import time

# np.random.seed(42)

mnist = fetch_mldata('MNIST original')
print(mnist)
X, y = mnist["data"], mnist["target"]
X, y = X.astype('float32'), y.astype('float32')

print('X shape: ', X.shape, X.dtype, '\nY shape: ', y.shape, y.dtype)

# Create test and training set. MNIST dataset already splitted up
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Shuffle the data
rnd_idx = np.random.permutation(len(X_train))
X_train = X_train[rnd_idx]
y_train = y_train[rnd_idx]

# Scaling data (mean 0, variance 1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape and apply OneHotEncoder to compute the 3class classifier
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
from sklearn.preprocessing import OneHotEncoder

onehot_encoder = OneHotEncoder(sparse=False)
y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)

# Convert data in HDF5 files
hpelm.make_hdf5(X_train, "hX_train.h5")
hpelm.make_hdf5(y_train, "hy_train.h5")
hpelm.make_hdf5(X_test, "hX_test.h5")
hpelm.make_hdf5(y_test, "hy_test.h5")

# Normalize does not work
# print(hpelm.normalize_hdf5('hX_train.h5'))
# print(hpelm.normalize_hdf5('hX_test.h5')

np.set_printoptions(precision=2)
neuron_number = 15000
out_class = 10
CV_folds = 10

# Problems when calling multiple files with same names Error: ValueError: The file 'hy_train_pred.h5' is already opened.  Please close it before reopening in write mode.
# Building model
print('\nBuilding model: HPELM with ')
model = hpelm.HPELM(X_train.shape[1], out_class, classification="c", batch=2048, accelerator="GPU", precision='single',
                    tprint=5)
model.add_neurons(neuron_number, 'sigm')
print(str(model))
t = time.time()
model.train('hX_train.h5', 'hy_train.h5', 'c')
elapsed_time_train = time.time() - t
print("Training time: %f" % elapsed_time_train)
model.predict('hX_train.h5', 'hy_train_pred.h5')
print('Training Accuracy: ', (1 - model.error('hy_train.h5', 'hy_train_pred.h5')))
model.predict('hX_test.h5', 'hy_test_pred.h5')
print('Test Accuracy: ', (1 - model.error('hy_test.h5', 'hy_test_pred.h5')))
