# Import dataset, from another files, splitted
import hpelm
import time
import numpy as np
from Misc.loadMNIST_orig import X_test, y_test, X_train, y_train
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape and apply OneHotEncoder to compute the 10class classifier
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False)
y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)

# Hyperparameters
np.set_printoptions(precision=2)
neuron_number = 4096
out_class = 10
CV_folds = 10
batch_size = neuron_number
# prec = "single"

print('\nELM Normal')
elm = hpelm.ELM(X_train.shape[1], out_class, classification="c",batch=batch_size, accelerator="GPU",
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


print('\nELM AUTOENCODER single')
auto_elm = hpelm.ELM(X_train.shape[1], X_train.shape[1], batch=batch_size, accelerator="GPU",
                precision='single')
auto_elm.add_neurons(8100, 'sigm')
print(str(auto_elm))
# Training model
t = time.time()
auto_elm.train(X_train, X_train)
elapsed_time_train = time.time() - t
y_train_predicted = auto_elm.predict(X_train)
print("Training time: %f" % elapsed_time_train)
print('Training Accuracy: ', (1 - auto_elm.error(X_train, y_train_predicted)))
# Prediction from trained model
y_test_predicted = auto_elm.predict(X_test)
print('Test Accuracy: ', (1 - auto_elm.error(X_test, y_test_predicted)))



