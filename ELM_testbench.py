# Import dataset, from another files, already scaled (StandardScaler) and splitted
import hpelm
import time
import numpy as np
from loadMNIST_orig import X_test, y_test, X_train, y_train

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Seed for random num gen, high dependence of accuracy from those numbers.
rnd_seed = 42
np.random.seed(42)

# Reshape and apply OneHotEncoder to compute the 3class classifier
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
from sklearn.preprocessing import OneHotEncoder

onehot_encoder = OneHotEncoder(sparse=False)
y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)

np.set_printoptions(precision=2)
neuron_number = 1024
out_class = 10
CV_folds = 10


# Building model
print('ELM')
elm = hpelm.ELM(X_train.shape[1],out_class, classification ="c")
elm.add_neurons(neuron_number,'sigm')
print(str(elm))
# Training model
t = time.time()
elm.train(X_train,y_train, 'c')
elapsed_time_train = time.time() - t
y_train_predicted = elm.predict(X_train)
print("Training time: %f" % elapsed_time_train)
print('Training Accuracy: ',(1-elm.error(y_train,y_train_predicted)))
# Prediction from trained model
y_test_predicted = elm.predict(X_test)
print('Test Accuracy: ',(1-elm.error(y_test,y_test_predicted)))
#print(elm.confusion(y_test,y_test_predicted)) #value as 4E+5
y_test_sk = y_test.argmax(1)
y_test_predicted_sk = y_test_predicted.argmax(1)
class_report = classification_report(y_test_sk, y_test_predicted_sk)
cnf_matrix = confusion_matrix(y_test_sk, y_test_predicted_sk)
print("Confusion Matrix:\n", cnf_matrix)
print("Classification report\n: ", class_report)



# LOO ELM
print('P-ELM')
elm = hpelm.ELM(X_train.shape[1],out_class, classification ="c")
elm.add_neurons(neuron_number,'sigm')
t = time.time()
elm.train(X_train,y_train,'LOO','c')
elapsed_time_train = time.time() - t
print(str(elm))
y_train_predicted = elm.predict(X_train)
print("Training time: %f" % elapsed_time_train)
print('Training Accuracy: ',(1-elm.error(y_train,y_train_predicted)))
y_test_predicted = elm.predict(X_test)
print('Test Accuracy: ',(1-elm.error(y_test,y_test_predicted)))
#print(elm.confusion(y_test,y_test_predicted)) #value as 4E+5
y_test_sk = y_test.argmax(1)
y_test_predicted_sk = y_test_predicted.argmax(1)
class_report = classification_report(y_test_sk, y_test_predicted_sk)
cnf_matrix = confusion_matrix(y_test_sk, y_test_predicted_sk)
print("Confusion Matrix:\n", cnf_matrix)
print("Classification report\n: ", class_report)


'''
# 'OP' (L1 reg) does not work, fuck!
# LOO ELM L1 Regularized
print('OP-ELM')
elm = hpelm.ELM(X_train.shape[1], out_class, classification="c")
elm.add_neurons(neuron_number, 'sigm')
t = time.time()
elm.train(X_train, y_train, 'LOO', 'OP', 'c')
elapsed_time_train = time.time() - t
print(str(elm))
y_train_predicted = elm.predict(X_train)
print("Training time: %f" % elapsed_time_train)
print('Training Accuracy: ', (1 - elm.error(y_train, y_train_predicted)))
y_test_predicted = elm.predict(X_test)
print('Test Accuracy: ',(1-elm.error(y_test,y_test_predicted)))
#print(elm.confusion(y_test,y_test_predicted)) #value as 4E+5
y_test_sk = y_test.argmax(1)
y_test_predicted_sk = y_test_predicted.argmax(1)
class_report = classification_report(y_test_sk, y_test_predicted_sk)
cnf_matrix = confusion_matrix(y_test_sk, y_test_predicted_sk)
print("Confusion Matrix:\n", cnf_matrix)
print("Classification report\n: ", class_report)
'''


# CV ELM 5 fold
print('CV-ELM, ', CV_folds,' fold')
elm = hpelm.ELM(X_train.shape[1], out_class, classification="c")
elm.add_neurons(neuron_number, 'sigm')
t = time.time()
e = elm.train(X_train, y_train, 'CV', 'c', k=CV_folds)
print('Error CV: ', e)
elapsed_time_train = time.time() - t
print(str(elm))
y_train_predicted = elm.predict(X_train)
print("Training time: %f" % elapsed_time_train)
print('Training Accuracy: ', (1 - elm.error(y_train, y_train_predicted)))
y_test_predicted = elm.predict(X_test)
print('Test Accuracy: ',(1-elm.error(y_test,y_test_predicted)))
#print(elm.confusion(y_test,y_test_predicted)) #value as 4E+5
y_test_sk = y_test.argmax(1)
y_test_predicted_sk = y_test_predicted.argmax(1)
class_report = classification_report(y_test_sk, y_test_predicted_sk)
cnf_matrix = confusion_matrix(y_test_sk, y_test_predicted_sk)
print("Confusion Matrix:\n", cnf_matrix)
print("Classification report\n: ", class_report)



# Validated ELM
print('Validate ELM')
from sklearn.model_selection import train_test_split
from loadMNIST_orig import X_test, y_test, X_train, y_train

# split train+validation set into training and validation sets
X_train, x_valid, y_train, y_valid = train_test_split(
    X_train, y_train, random_state=1)
print("Size of training set: {}   size of validation set: {}   size of test set:"
      " {}\n".format(X_train.shape[0], x_valid.shape[0], X_test.shape[0]))

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_valid = y_valid.reshape(-1, 1)
y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)
y_valid = onehot_encoder.fit_transform(y_valid)

elm = hpelm.ELM(X_train.shape[1], out_class, classification="c")
elm.add_neurons(neuron_number, 'sigm')
t = time.time()
elm.train(X_train, y_train, 'V', 'c', Xv=x_valid, Tv=y_valid)
elapsed_time_train = time.time() - t
print(str(elm))
y_train_predicted = elm.predict(X_train)
print("Training time: %f" % elapsed_time_train)
print('Training Accuracy: ', (1 - elm.error(y_train, y_train_predicted)))
y_test_predicted = elm.predict(X_test)
print('Test Accuracy: ',(1-elm.error(y_test,y_test_predicted)))
#print(elm.confusion(y_test,y_test_predicted)) #value as 4E+5
y_test_sk = y_test.argmax(1)
y_test_predicted_sk = y_test_predicted.argmax(1)
class_report = classification_report(y_test_sk, y_test_predicted_sk)
cnf_matrix = confusion_matrix(y_test_sk, y_test_predicted_sk)
print("Confusion Matrix:\n", cnf_matrix)
print("Classification report\n: ", class_report)
