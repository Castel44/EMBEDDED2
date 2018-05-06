import ELM.stl10_input as stl10
from ELM.stl10_input import DATA_PATH, LABEL_PATH
import hpelm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import time
import matplotlib.pyplot as plt


def histogram_labels(y_train, y_test):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

    fig.suptitle('Class Distribution', fontsize=14, fontweight='bold', y=1.05)

    ax1.hist(y_train, bins=10)
    ax1.set_title("Training set")
    ax1.set_xlim(0, 9)

    ax2.hist(y_test, color='g', bins=10)
    ax2.set_title("Test set")

    fig.tight_layout()

    plt.show()

###############################################################################################################
# load dataset
print('STL10 DATASET')
data = stl10.read_all_images(DATA_PATH)
labels = stl10.read_labels(LABEL_PATH)


x_train, x_test, y_train, y_test = train_test_split(data.astype('float32'), labels.astype('float32'), shuffle=True,
                                                    test_size=0.2)

print('x_train shape ', x_train.shape)
print('y_train shape ', y_train.shape)
print('x_test shape ', x_test.shape)
print('y_test shape ', y_test.shape)

# plot histogram of labels
histogram_labels(y_train, y_test)
# plot dataset example
stl10.plot_example(data, labels)

print('Reshape and scaling data (mean 0, std 1)')
x_train = x_train.reshape(
    (len(x_train), x_train.shape[1] * x_train.shape[2] * x_train.shape[3]))
x_test = x_test.reshape((len(x_test), x_test.shape[1] * x_test.shape[2] * x_test.shape[3]))

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
onehot_encoder = OneHotEncoder(sparse=False)
y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.transform(y_test)

###############################################################################################################
#HYPERPARAMETERS
neuron_number = 4096
out_class = 10

print('ELM-GPU single')
elm = hpelm.ELM(x_train.shape[1], out_class, classification="c", batch=1000, accelerator="GPU",
                precision='single', norm=10**-3)
elm.add_neurons(neuron_number, 'sigm')
# elm.add_neurons(x_train.shape[1],'lin')
print(str(elm))
# Training model
t = time.time()
elm.train(x_train, y_train, 'c')
elapsed_time_train = time.time() - t
y_train_predicted = elm.predict(x_train)
print("Training time: %f" % elapsed_time_train)
print('Training Accuracy: ', (1 - elm.error(y_train, y_train_predicted)))
# Prediction from trained model
y_test_predicted = elm.predict(x_test)
print('Test Accuracy: ', (1 - elm.error(y_test, y_test_predicted)))


