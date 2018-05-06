import time
import hpelm
import numpy as np
import random
import h5py
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from CIFAR10.cifar10dataset import train_data, train_labels, test_data, test_labels
from keras.preprocessing.image import ImageDataGenerator
from CIFAR10.augmented_data import crop

print("Loading Dataset: CIFAR10")
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


#######################################################################################################################
# Data Preprocess and augmention
print('Reshape y and apply OneHotEncoder for %d class' % out_class)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
onehot_encoder = OneHotEncoder(sparse=False)
y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)

# print('Crop imago to 24x24, centrally for eval and randomly for training')
# X_train = crop(X_train,dim= 24, rand= True)
# X_test = crop(X_test, dim=24, rand= False)

print('Data augmentation with keras.image')
gen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    zca_whitening=False,
    shear_range=0.2,
    channel_shift_range=0.2,
    fill_mode='nearest',
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=False,
    vertical_flip=False,
    data_format='channels_last'
)

# compute quantities required for featurewise normalization (std, mean, and principal components if ZCA whitening is applied)
# gen.fit(X_train)

# AUG Hyperparameters
img_batches = 2000
epoch = 6
t_print = 10  #time to display eta

X_train_aug = X_train
y_train_aug = y_train

batches = 0
t = time.time()
t0 = time.time()
num_batches = (len(X_train) // img_batches) * epoch
for X_batch, y_batch in gen.flow(X_train, y_train, batch_size=img_batches):
    X_train_aug = np.append(X_train_aug, X_batch, axis=0)
    y_train_aug = np.append(y_train_aug, y_batch, axis=0)

    # report time
    eta = int(((time.time() - t0) / (batches + 1)) * (num_batches - batches - 1))
    if time.time() - t > t_print:
        print('processing image batch %d/%d, eta: %d:%02d:%02d' % (
            batches, num_batches, eta / 3600, (eta % 3600) / 60, eta % 60))
        t = time.time()

    batches += 1
    if batches >= num_batches:
        break


print('CIFAR 10 DATASET agumented')
print('X_train shape ', X_train_aug.shape)
print('y_train shape ', y_train_aug.shape)
print('X_test shape ', X_test.shape)
print('y_test shape ', y_test.shape)

print('X_train mean: ', X_train_aug.mean())
print('X_train std: ', X_train_aug.std())
#######################################################################################################################


print('Reshape data, X.shape [num_istance, features]')
X_train_aug = X_train_aug.reshape(
    (len(X_train_aug), X_train_aug.shape[1] * X_train_aug.shape[2] * X_train_aug.shape[3]))
X_test = X_test.reshape((len(X_test), X_test.shape[1] * X_test.shape[2] * X_test.shape[3]))

print('Scale data, mean= 0, std= 1')
scaler = StandardScaler()
X_train_aug = scaler.fit_transform(X_train_aug)
X_test = scaler.transform(X_test)
print('X_train mean: ', X_train_aug.mean())
print('X_train std: ', X_train_aug.std())

# Convert data in HDF5 files
print('Create HDF5 data')
hpelm.make_hdf5(X_train_aug, "hX_train.h5")
hpelm.make_hdf5(y_train_aug, "hy_train.h5")
hpelm.make_hdf5(X_test, "hX_test.h5")
hpelm.make_hdf5(y_test, "hy_test.h5")

#######################################################################################################################
# HYPERPARAMETERS
np.set_printoptions(precision=2)
neuron_number = 10000
CV_folds = 10
batch_size = neuron_number
# np.random.seed(42)
neuron_type = ('tanh', 'sigm')
n_estimator = 10

#######################################################################################################################
# Training networks
print()
print("Bagging %d ELM classificator\n" % n_estimator)
path_file = 'C:\\Users\\Andrea\\PycharmProjects\\EMBEDDED\\CIFAR10\\'
train_pred_str = 'hy_train_pred{}.h5'
test_pred_str = 'hy_test_pred{}.h5'
acc_train = []
acc_test = []
pred_labels = []
tstart = time.time()
for i in range(n_estimator):
    print('Estim #%d' % i)
    model = hpelm.HPELM(3072, 10, classification="c", batch=batch_size, accelerator="GPU", precision='single',
                        tprint=5)
    model.add_neurons(neuron_number, random.choice(neuron_type))
    print(str(model))
    t = time.time()
    model.train('hX_train.h5', 'hy_train.h5', 'c')
    elapsed_time_train = time.time() - t
    print("Training time: %f" % elapsed_time_train)

    model.predict('hX_train.h5', train_pred_str.format(i + 1))
    acc_train.append(1 - model.error('hy_train.h5', train_pred_str.format(i + 1)))
    print('Training Accuracy: ', acc_train[i])
    model.predict('hX_test.h5', test_pred_str.format(i + 1))
    acc_test.append(1 - model.error('hy_test.h5', test_pred_str.format(i + 1)))
    print('Test Accuracy: ', acc_test[i])

    hf = h5py.File(path_file + test_pred_str.format(1 + i))
    data = hf.get('data').value
    pred_labels.append(data)
    hf.close()

print("Computation done in ", (time.time() - tstart), "seconds!!")
print("###############################################################################################\n")

acc_train = np.array(acc_train)
acc_test = np.array(acc_test)
pred_labels = np.array(pred_labels)

# Print recap of networks
for i in range(n_estimator):
    msg = "Network: {0}, Accuracy on Training-Set: {1:.6f}, Test-Set: {2:.6f}"
    print(msg.format(i, acc_train[i], acc_test[i]))

print("\nMean test-set accuracy: {0:.4f}".format(np.mean(acc_test)))
print("Min test-set accuracy:  {0:.4f}".format(np.min(acc_test)))
print("Max test-set accuracy:  {0:.4f}".format(np.max(acc_test)))

# building ensemble
ensemble_pred_labels = np.mean(pred_labels, axis=0)
ensemble_cls_pred = np.argmax(ensemble_pred_labels, axis=1)
ensemble_correct = (ensemble_cls_pred == test_labels)
ensemble_incorrect = np.logical_not(ensemble_correct)

# best network
best_net = np.argmax(acc_test)
best_net_pred_labels = pred_labels[best_net, :, :]
best_net_cls_pred = np.argmax(best_net_pred_labels, axis=1)
best_net_correct = (best_net_cls_pred == test_labels)
best_net_incorrect = np.logical_not(best_net_correct)

# Ensemble and Best network comparison
print("\nBest Net correct estimated instances: ", np.sum(best_net_correct))
print("Ensamble correct estimated instances: ", np.sum(ensemble_correct))

ensemble_better = np.logical_and(best_net_incorrect, ensemble_correct)
best_net_better = np.logical_and(best_net_correct, ensemble_incorrect)
print("Best Net better classification: ", best_net_better.sum())
print("Ensemble better classification: ", ensemble_better.sum())

ensemble_acc = (ensemble_cls_pred == test_labels).mean()
best_net_acc = acc_test[best_net]
print("\nEnsemble accuracy: ", ensemble_acc * 100)
print("Best net accuracy: ", acc_test[best_net] * 100)

