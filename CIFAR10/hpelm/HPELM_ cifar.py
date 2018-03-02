import time

import hpelm
import numpy as np

'''
X_train = train_data.astype('float32')
y_train = train_labels.astype('float32')
X_test = test_data.astype('float32')
y_test = test_labels.astype('float32')

# Reshape and scale
X_train = X_train.reshape((len(X_train),X_train.shape[1]*X_train.shape[2]*X_train.shape[3]))
X_test = X_test.reshape((len(X_test),X_test.shape[1]*X_test.shape[2]*X_test.shape[3]))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
onehot_encoder = OneHotEncoder(sparse=False)
y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)


# Convert data in greyscale
#X_train = rgb2gray(X_train)
#X_test = rgb2gray(X_test)

print('CIFAR 10 DATASET')
print('X_train shape ', X_train.shape)
print('y_train shape ', y_train.shape)
print('X_test shape ', X_test.shape)
print('y_test shape ', y_test.shape)
#out_class = len(np.unique(y_test))
#print('Num Classes: ', out_class)


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


# Concatenate input tensor
X_train = np.concatenate((X_train,X_train), axis=0)
y_train = np.concatenate((y_train, y_train), axis=0)
X_train = np.concatenate((X_train,X_train), axis=0)
y_train = np.concatenate((y_train, y_train), axis=0)
X_train = np.concatenate((X_train,X_train), axis=0)
y_train = np.concatenate((y_train, y_train), axis=0)

rnd_idx = np.random.permutation(len(X_train))
X_train = X_train[rnd_idx]
y_train = y_train[rnd_idx]

print('CIFAR 10 DATASET agumented')
print('X_train shape ', X_train.shape)
print('y_train shape ', y_train.shape)
print('X_test shape ', X_test.shape)
print('y_test shape ', y_test.shape)



print('Reshape and scaling data (mean 0, std 1)')
X_train = X_train.reshape((len(X_train),X_train.shape[1]*X_train.shape[2]*X_train.shape[3])) #X_train.shape[3] if coloured
X_test = X_test.reshape((len(X_test),X_test.shape[1]*X_test.shape[2]*X_test.shape[3]))

#X_train = X_train/X_train.std(0)
#X_test = X_test/X_test.std(0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


print('Reshape and apply OneHotEncoder')
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
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
'''

np.set_printoptions(precision=2)
neuron_number = 10240
CV_folds = 10

# Problems when calling multiple files with same names Error: ValueError: The file 'hy_train_pred.h5' is already opened.  Please close it before reopening in write mode.
# Building model
print('\nBuilding model: HPELM with ')
model = hpelm.HPELM(3072, 10, classification="c", batch=512, accelerator="GPU", precision='single',
                    tprint=5)
model.add_neurons(neuron_number, 'sigm')
# model.add_neurons(500, 'lin')
print(str(model))
t = time.time()
model.train('hX_train.h5', 'hy_train.h5', 'c')
elapsed_time_train = time.time() - t
print("Training time: %f" % elapsed_time_train)
model.predict('hX_train.h5', 'hy_train_pred.h5')
print('Training Accuracy: ', (1 - model.error('hy_train.h5', 'hy_train_pred.h5')))
model.predict('hX_test.h5', 'hy_test_pred.h5')
print('Test Accuracy: ', (1 - model.error('hy_test.h5', 'hy_test_pred.h5')))
