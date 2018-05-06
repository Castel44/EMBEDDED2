# Import dataset, from another files, splitted
import hpelm
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


def plot_images(instances, images_per_row=20, size=28, **options):
    plt.figure()
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row: (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap='binary', **options)
    plt.axis("off")
    plt.show()


def subset(X, y, num_sample=20):
    idx = np.array(np.where(y == 0))
    idx = idx.reshape(idx.shape[1], )
    X_train_reduced = X[np.random.choice(idx, num_sample, replace=False)]
    y_train_reduced = y[np.random.choice(idx, num_sample, replace=False)]
    for i in range(9):
        idx = np.array(np.where(y == i + 1))
        idx = idx.reshape(idx.shape[1], )
        X_train_reduced = np.append(X_train_reduced, X[np.random.choice(idx, num_sample, replace=False)], axis=0)
        y_train_reduced = np.append(y_train_reduced, y[np.random.choice(idx, num_sample, replace=False)], axis=0)
    return X_train_reduced, y_train_reduced


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


# Load Dataset
mnist = fetch_mldata('MNIST original')
print(mnist)
X, y = mnist["data"], mnist["target"]
X, y = X.astype('float32'), y.astype('float32')
print('X shape: ', X.shape, X.dtype, '\nY shape: ', y.shape, y.dtype)

# Create test and training set. MNIST dataset already splitted up
X_tr, X_te, y_tr, y_te = X[:60000], X[60000:], y[:60000], y[60000:]

# Create subset of MINST datset
# X_train, y_train = subset(X_train, y_train)

# Hyperparameters
np.set_printoptions(precision=2)
# np.random.seed(42)
neuron_number = 20
out_class = 10
batch_size = neuron_number

B = []
B_raw = []
orth_init = False
for i in range(10):
    X_train = X_tr[y_tr == i]

    # Shuffle the data
    rnd_idx = np.random.permutation(len(X_train))
    X_train = X_train[rnd_idx]

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    print('\nELM-AE')
    auto_elm = hpelm.ELM(X_train.shape[1], X_train.shape[1], batch=batch_size, accelerator="GPU",
                         precision='single')
    if orth_init:
        Bias, W = ortho_B_W(neuron_number, X_train.shape[1])
        auto_elm.add_neurons(neuron_number, 'sigm', B=np.asarray(Bias, dtype=np.float32, order='C'),
                             W=np.asarray(W, dtype=np.float32, order='C'))  # use for ortho B and W  init
    else:
        auto_elm.add_neurons(neuron_number, 'sigm')  #Use for random B/W
    print(str(auto_elm))
    # Training model
    t = time.time()
    auto_elm.train(X_train, X_train)
    elapsed_time_train = time.time() - t
    print("Training time: %f" % elapsed_time_train)
    B.append(scaler.inverse_transform(auto_elm.nnet.get_B()))
    B_raw.append(auto_elm.nnet.get_B())


B = np.array(B).reshape(neuron_number * 10, -1)
B_raw = np.array(B_raw).reshape(neuron_number * 10, -1)
plot_images(B, images_per_row=20)
