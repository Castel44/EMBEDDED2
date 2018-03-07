# Import dataset, from another files, already scaled (StandardScaler) and splitted
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

# w = 1 / X_train.shape[0] * np.ones((out_class, 1))

print('\nELM AUTOENCODER single')
elm = hpelm.ELM(X_train.shape[1], X_train.shape[1], batch=batch_size, accelerator="GPU",
                precision='single')
elm.add_neurons(neuron_number, 'sigm')
# elm.add_neurons(X_train.shape[1],'lin')
print(str(elm))
# Training model
t = time.time()
elm.train(X_train, X_train)
elapsed_time_train = time.time() - t
y_train_predicted = elm.predict(X_train)
print("Training time: %f" % elapsed_time_train)
print('Training Accuracy: ', (1 - elm.error(X_train, y_train_predicted)))
# Prediction from trained model
y_test_predicted = elm.predict(X_test)
print('Test Accuracy: ', (1 - elm.error(X_test, y_test_predicted)))
# print(elm.confusion(y_test,y_test_predicted)) #value as 4E+5
# y_test_sk = y_test.argmax(1)
# y_test_predicted_sk = y_test_predicted.argmax(1)
# class_report = classification_report(y_test_sk, y_test_predicted_sk)
# cnf_matrix = confusion_matrix(y_test_sk, y_test_predicted_sk)
# print("Confusion Matrix:\n", cnf_matrix)
# print("Classification report\n: ", class_report)
