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
X, y = X.astype('float32'), y.astype('float32')

print('X shape: ', X.shape, X.dtype, '\nY shape: ', y.shape, y.dtype)

#Create test and training set. MNIST dataset already splitted up
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

'''
# PCA for visualize data, not great, but blazing fast
from sklearn.decomposition import PCA
# Create a Randomized PCA model that takes two components
randomized_pca = PCA(n_components=2, random_state=42)
# Fit and transform the data to the model reduced_data_rpca = randomized_pca.fit_transform(digits.data)
# Create a regular PCA model pca = PCA(n_components=2)
# Fit and transform the data to the model
reduced_data_rpca = randomized_pca.fit_transform(X_train)
print('Type reduced data: ',type(reduced_data_rpca))
# Inspect the shape
print('Shape reduced data; ',reduced_data_rpca.shape)
# reduced to 2D

colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
for i in range(len(colors)):
     x = reduced_data_rpca[:, 0][y_train == i]
     y = reduced_data_rpca[:, 1][y_train == i]

     plt.scatter(x, y, c=colors[i])
plt.legend(np.linspace(0,9,10), bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) #target names = linspace
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title("PCA Scatter Plot")
plt.show()
'''

#Shuffle the data
# np.random.seed(42)
rnd_idx = np.random.permutation(len(X_train))
X_train = X_train[rnd_idx]
y_train = y_train[rnd_idx]

'''
#Scaling data (mean 0, variance 1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
'''
