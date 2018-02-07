# Common imports
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import time

from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


mnist = fetch_mldata('MNIST original')
print(mnist)
X, y = mnist["data"], mnist["target"]
print('X shape \n',X.shape, '\n Y shape \n', y.shape)

some_digit = X[36000]
'''
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,
           interpolation="nearest")
plt.axis("off")
plt.show()
print('Label of plotted figure',y[36000])

def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")

# EXTRA
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")

plt.figure(figsize=(9,9))
example_images = np.r_[X[:12000:600], X[13000:30600:600], X[30600:60000:590]]
plot_digits(example_images, images_per_row=10)
plt.show()
'''

#Create test and training set. MNIST dataset already splitted up
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

np.random.seed(42)
rnd_idx = np.random.permutation(60000)
X_train = X_train[rnd_idx]
y_train = y_train[rnd_idx]

#Scaling data (mean 0, variance 1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Train model and save time
sgd_clf = SGDClassifier(random_state=42)
t = time.time()
sgd_clf.fit(X_train, y_train)
elapsed_time_train = time.time() - t

#Predict value
t = time.time();
Y_prediction = sgd_clf.predict(X_test)
elapsed_time_test = time.time() - t

print('SDGClassifier\n')
print("Training time: %f" % elapsed_time_train)
print("Testing time: %f" % elapsed_time_test)

acc_score = accuracy_score(y_test, Y_prediction)
cnf_matrix = confusion_matrix(y_test, Y_prediction)
class_report = classification_report(y_test, Y_prediction)
np.set_printoptions(precision=2)
print("Accuracy:\n", acc_score)
print("Confusion Matrix:\n", cnf_matrix)
print("Classification report\n: ", class_report)

print('FOREST\n')
forest_clf = RandomForestClassifier(random_state=42)
t = time.time()
forest_clf.fit(X_train, y_train)
elapsed_time_train = time.time() - t

#Predict value
t = time.time();
Y_prediction = forest_clf.predict(X_test)
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