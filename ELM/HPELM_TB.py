from ELM.hpelm_testbench import load_mnist, load_cifar
import hpelm
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools

# Get dataset
x_train, x_test, y_train, y_test, img_size, img_channels = load_mnist()

# Data scaler
from sklearn.preprocessing import StandardScaler

prescaler = StandardScaler()
x_train = prescaler.fit_transform(x_train)
x_test = prescaler.transform(x_test)

# Hyperparameters
input_size = img_size ** 2 * img_channels
output_size = 10
n_neurons = (100, 100, 500, 1000, 5000, 8000, 15000)
batch_size = 1000
n_epochs = 1
reps = 3
norm = reps*(None,)

# Convert data in HDF5 files
print('Create HDF5 data')
hpelm.make_hdf5(x_train, "hX_train.h5")
hpelm.make_hdf5(y_train, "hy_train.h5")
hpelm.make_hdf5(x_test, "hX_test.h5")
hpelm.make_hdf5(y_test, "hy_test.h5")

train_time = []
train_acc = []
test_acc = []
run = 0
run_comb = list(itertools.product(n_neurons, norm))
for v in itertools.product(n_neurons, norm):
    print('\nStarting run %d/%d' % (run + 1, run_comb.__len__()))
    print('Hyperpar: neurons= ', v[0], 'norm=', v[1])
    model = hpelm.HPELM(input_size, output_size, classification="c", batch=batch_size, accelerator="GPU",
        precision='single',
        tprint=2, norm=v[1])
    model.add_neurons(v[0], 'sigm')
    #print(str(model))
    t = time.time()
    model.train('hX_train.h5', 'hy_train.h5', 'c')
    train_time.append(time.time() - t)
    print("Training time: %f" % train_time[run])
    model.predict('hX_train.h5', 'hy_train_pred.h5')
    train_acc.append(1 - model.error('hy_train.h5', 'hy_train_pred.h5'))
    print('Training Accuracy: ', train_acc[run])
    model.predict('hX_test.h5', 'hy_test_pred.h5')
    test_acc.append(1 - model.error('hy_test.h5', 'hy_test_pred.h5'))
    print('Test Accuracy: ', test_acc[run])
    model.nnet.reset()  # free GPU memory
    model.__del__()  # close opened h5 files
    run += 1

print('\nDone training!')
# Searching for best hypar combination
best_net = np.argmax(test_acc)
print('Best net with hypepar:')
print('  -neuron number:', run_comb[best_net][0])
print('  -norm:', run_comb[best_net][1])
print('Best net test accuracy: ', test_acc[best_net])

print('Train Time')
for i in range(7):
    print(np.array(train_time[i*reps:i*reps+reps]).mean())

