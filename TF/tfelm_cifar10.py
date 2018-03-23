import numpy as np
from TF.elm import elm
import tensorflow as tf
import itertools
from keras.datasets import cifar10
from keras.utils import to_categorical as OneHot
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

print("Loading Dataset: CIFAR10")
# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = OneHot(y_train, 10)
y_test = OneHot(y_test, 10)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print('Reshape data, X.shape [num_instance, features]')
x_train = x_train.reshape(
    (len(x_train), x_train.shape[1] * x_train.shape[2] * x_train.shape[3]))
x_test = x_test.reshape((len(x_test), x_test.shape[1] * x_test.shape[2] * x_test.shape[3]))

print('Scale data, mean= 0, std= 1')
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

######################################################################################################################
# Hyperparameters
input_size = x_train.shape[1]
output_size = 10
n_neurons = 5000
batch_size = 5000
norm = (None, 10 ** -3, 10 ** 0, 10 ** 2, 10 ** 4,)
ortho_w = tf.orthogonal_initializer()
unit_b = tf.uniform_unit_scaling_initializer()
init = ((None, None), (ortho_w, unit_b),)

train_acc = []
test_acc = []
run = 0
run_comb = list(itertools.product(init, norm))
for v in itertools.product(init, norm):
    print('\nStarting run %d/%d' % (run + 1, run_comb.__len__()))
    model = elm(input_size, output_size, n_neurons, w_initializer=v[0][0], b_initializer=v[0][1],
                l2norm=v[1], batch_size=batch_size)
    train_acc.append(model.train(x_train, y_train))
    test_acc.append(model.evaluate(x_test, y_test))
    print('Test accuracy: ', test_acc[run])
    del model
    run += 1

print('Done training!')
# os.system('tensorboard --logdir=%s' % savedir)

# Searching for best hypar combination
best_net = np.argmax(test_acc)
print('Best net with hypepar:')
print('  -neuron number: ', run_comb[best_net][0])
print('  -norm: 10 **', run_comb[best_net][1])
print('Best net test accuracy: ', test_acc[best_net])

plt.semilogx(norm, train_acc, 'b', label='Train Accuracy')
plt.semilogx(norm, test_acc, 'g', label='Test Accuracy')
plt.plot(norm, train_acc, 'b*')
plt.plot(norm, test_acc, 'g*')
plt.xlabel('Normalization coefficient')
plt.ylabel('Accuracy')
plt.title('Cifar10 ELM, 15k random sigmoid neuron, no data aug')
plt.grid()
plt.legend()
