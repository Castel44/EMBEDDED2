import numpy as np
import hpelm
import itertools
import time
import os

# SVHN scaled dataset in ./data/ ---
# already in hdf5 format

path = os.getcwd() + '/data/'

# Hyperpar
batch_size = 5000
neuron_number = (8196, 8196*2, 8196*3,)
norm = (None,)

test_acc = []
val_acc = []
run = 0
run_comb = list(itertools.product(neuron_number, norm))
for v in itertools.product(neuron_number, norm):
    print('\nStarting run %d/%d' % (run + 1, run_comb.__len__()))
    print('Hyperpar: neuron_number=', v[0], 'norm=', v[1])
    model = hpelm.HPELM(32*32*3, 10, classification="c", batch=batch_size, accelerator="GPU",
                        precision='single', tprint=5, norm=v[1])
    model.add_neurons(v[0], 'sigm')
    print(str(model))
    t = time.time()
    model.train(path + 'hX_train.h5', path + 'hy_train.h5', 'c')
    elapsed_time_train = time.time() - t
    print("Training time: %f" % elapsed_time_train)
    model.predict(path + 'hX_test.h5', path + 'hy_test_pred.h5')
    test_acc.append(1 - model.error(path + 'hy_test.h5',path +  'hy_test_pred.h5'))
    print('Test Accuracy: ', test_acc[run])
    model.predict(path + 'hX_val.h5', path + 'hy_val_pred.h5')
    val_acc.append(1 - model.error(path + 'hy_val.h5', path + 'hy_val_pred.h5'))
    print('Val Accuracy: ', val_acc[run])
    model.nnet.reset()  # free GPU memory
    model.__del__()  # close opened h5 files
    run += 1

print('Done training!')

# Searching for best hypar combination
best_net = np.argmax(test_acc)
print('Best net with hypepar:')
print('  -neuron number:', run_comb[best_net][0])
print('  -norm:', run_comb[best_net][1])
print('Best net test accuracy: ', test_acc[best_net])


import matplotlib.pyplot as plt
'''
plt.semilogx([10**-6,10**-3,10],test_acc[:3],label='4096')
plt.semilogx([10**-6,10**-3,10],test_acc[3:],label='8196')
plt.plot(neuron_number ,test_acc, '*')
plt.plot([10**-6,10**-3,10],test_acc[3:], '*')
plt.grid()
plt.legend()
'''

plt.plot(neuron_number ,test_acc, '*')
plt.ylabel('Test Accuracy')
plt.xlabel('Neuron Number')
plt.title('SVHN dataset (data scaled, mean0, std1)')
plt.grid()