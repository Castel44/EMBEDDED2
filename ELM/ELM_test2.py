import hpelm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import time
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection

digits_sk = datasets.load_digits()
print(digits_sk.data.shape)

digits_tra = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra",
                         header=None)
print(digits_tra.shape)
digits_tes = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes",
                         header=None)
print(digits_tes.shape)

x_train = digits_tra.values[:, 0:64]
y_train = digits_tra.values[:, 64]
print(x_train.shape, y_train.shape)
x_test = digits_tes.values[:, 0:64]
y_test = digits_tes.values[:, 64]
print(x_test.shape, y_test.shape)

# x_train, x_test, y_train, y_test = model_selection.train_test_split(digits_sk.data,digits_sk.target, test_size=0.25, random_state=42)

'''
#Plot a digit
some_digit = x_train[1000]

def plot_digit(data):
    image = data.reshape(8, 8)
    plt.imshow(image, cmap = plt.cm.binary,
               interpolation="nearest")
    plt.axis("off")
    plt.show()

plot_digit(some_digit)
print('Label of plotted figure',y_train[1000])

# Plot some digit
# Figure size (width, height) in inches
fig = plt.figure(figsize=(6, 6))

# Adjust the subplots
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# For each of the 64 images
for i in range(64):
    # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    # Display an image at the i-th position
    ax.imshow(x_train[i].reshape(8,8), cmap=plt.cm.binary, interpolation='nearest')
    # label the image with the target value
    ax.text(0, 7, str(y_train[i]))

# Show the plot
plt.show()
'''

# Shuffle data
np.random.seed(42)
rnd_idx = np.random.permutation(3823)
x_train = x_train[rnd_idx]
y_train = y_train[rnd_idx]

# Scaling data (mean 0, variance 1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Test some models
sgd_clf = SGDClassifier(random_state=42)
t = time.time()
sgd_clf.fit(x_train, y_train)
elapsed_time_train = time.time() - t
# Predict value
t = time.time()
Y_prediction = sgd_clf.predict(x_test)
elapsed_time_test = time.time() - t
print('SDGClassifier')
print("Training time: %f" % elapsed_time_train)
print("Testing time: %f" % elapsed_time_test)
acc_score = accuracy_score(y_test, Y_prediction)
cnf_matrix = confusion_matrix(y_test, Y_prediction)
class_report = classification_report(y_test, Y_prediction)
np.set_printoptions(precision=2)
print("Accuracy:\n", acc_score)
print("Confusion Matrix:\n", cnf_matrix)
print("Classification report\n: ", class_report)

print('FOREST')
forest_clf = RandomForestClassifier(random_state=42)
t = time.time()
forest_clf.fit(x_train, y_train)
elapsed_time_train = time.time() - t
# Predict value
t = time.time();
Y_prediction = forest_clf.predict(x_test)
elapsed_time_test = time.time() - t
print("Training time: %f" % elapsed_time_train)
print("Testing time: %f" % elapsed_time_test)
acc_score = accuracy_score(y_test, Y_prediction)
cnf_matrix = confusion_matrix(y_test, Y_prediction)
class_report = classification_report(y_test, Y_prediction)
np.set_printoptions(precision=2)
print("Accuracy:\n", acc_score)
print("Confusion Matrix:\n", cnf_matrix)
print("Classification report\n: ", class_report)

# Reshape and apply OneHotEncoder to compute the 10class classifier
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
from sklearn.preprocessing import OneHotEncoder

onehot_encoder = OneHotEncoder(sparse=False)
y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)

print('ELM\n')
model = hpelm.ELM(64, 10, classification="c")
model.add_neurons(500, 'sigm')
model.add_neurons(64, 'lin')
print(str(model))
t = time.time()
model.train(x_train, y_train)
elapsed_time_train = time.time() - t
y_train_predicted = model.predict(x_train)

print("Training time: %f" % elapsed_time_train)
print('Training Error: ', model.error(y_train, y_train_predicted))
y_test_predicted = model.predict(x_test)
print('Test Error: ', model.error(y_test, y_test_predicted))
# PORCODDIO NON FUNZIONA LA CONFUSION MATRIX SU STO TOOLBOX DI MERDA
# print(model.confusion(y_test,y_test_predicted))

y_test_sk = y_test.argmax(1)
y_test_predicted_sk = y_test_predicted.argmax(1)
acc_score = accuracy_score(y_test_sk, y_test_predicted_sk)
cnf_matrix = confusion_matrix(y_test_sk, y_test_predicted_sk)
class_report = classification_report(y_test_sk, y_test_predicted_sk)
np.set_printoptions(precision=2)
print("Accuracy:\n", acc_score)
print("Confusion Matrix:\n", cnf_matrix)
print("Classification report\n: ", class_report)

print('ELM CV')
t = time.time()
model.train(x_train, y_train, 'CV', k=100)
elapsed_time_train = time.time() - t
print(str(model))
y_train_predicted = model.predict(x_train)

print("Training time: %f" % elapsed_time_train)
print('Training Error: ', model.error(y_train, y_train_predicted))
y_test_predicted = model.predict(x_test)
print('Test Error: ', model.error(y_test, y_test_predicted))

y_test_sk = y_test.argmax(1)
y_test_predicted_sk = y_test_predicted.argmax(1)
acc_score = accuracy_score(y_test_sk, y_test_predicted_sk)
cnf_matrix = confusion_matrix(y_test_sk, y_test_predicted_sk)
class_report = classification_report(y_test_sk, y_test_predicted_sk)
np.set_printoptions(precision=2)
print("Accuracy:\n", acc_score)
print("Confusion Matrix:\n", cnf_matrix)
print("Classification report\n: ", class_report)

'''
t = time.time()
model.train(x_train,y_train, 'LOO', 'OP')
elapsed_time_train = time.time() - t
print(str(model))
y_train_predicted = model.predict(x_train)
print('ELM')
print("Training time: %f" % elapsed_time_train)
print('Training Error: ',model.error(y_train,y_train_predicted))
y_test_predicted = model.predict(x_test)
print('Test Error: ',model.error(y_test,y_test_predicted))
'''
