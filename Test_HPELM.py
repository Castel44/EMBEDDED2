import hpelm as hp
import numpy as np
import matplotlib.pyplot as plt

# method 1
#load mnist digits data from sklearn
from sklearn import datasets
digits = datasets.load_digits()
# note: data is already preprocessed 32x32 => 8x8 reduced dimensionality

# method 2
#load mnist using panda
#import pandas as pd
#digits = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra", header=None)


type(digits)
print(digits.data.shape)
# 1797 images each 8x8 BW pixels
type(digits.data)
print(digits.DESCR)

# hpelm toolkit should take in input also numpy arrays

# this will be equal to the number of input neurons used. INSTANCE BASED LEARNING FOR NOW
in_size = digits.data.shape[1] #64 attributes



#plot an image from MNIST
plt.gray()
plt.matshow(digits.images[34])
plt.show()

# show more info on data
print(digits.target.shape)
# labels
digits.keys()

np.unique(digits.target)
out_size = len(np.unique(digits.target))
print(out_size)

digits.data.shape
digits.images.shape
print(np.all(digits.images.reshape((1797,64)) == digits.data))
#same data


# Import matplotlib
import matplotlib.pyplot as plt

# Figure size (width, height) in inches
fig = plt.figure(figsize=(6, 6))

# Adjust the subplots
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# For each of the 64 images
for i in range(64):
    # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    # Display an image at the i-th position
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    # label the image with the target value
    ax.text(0, 7, str(digits.target[i]))

# Show the plot
plt.show()

# Join the images and target labels in a list
images_and_labels = list(zip(digits.images, digits.target))

# for every element in the list
for index, (image, label) in enumerate(images_and_labels[:8]):
    # initialize a subplot of 2X4 at the i+1-th position
    plt.subplot(2, 4, index + 1)
    # Don't plot any axes
    plt.axis('off')
    # Display images in all subplots
    plt.imshow(image, cmap=plt.cm.gray_r,interpolation='nearest')
    # Add a title to each subplot
    plt.title('Training: ' + str(label))

# Show the plot
plt.show()

from sklearn import decomposition

# Create a Randomized PCA model that takes two components
randomized_pca = decomposition.PCA(n_components=2)

# Fit and transform the data to the model reduced_data_rpca = randomized_pca.fit_transform(digits.data)

# Create a regular PCA model pca = PCA(n_components=2)

# Fit and transform the data to the model
reduced_data_rpca = randomized_pca.fit_transform(digits.data)
type(reduced_data_rpca)
# Inspect the shape
reduced_data_rpca.shape
# reduced to 2D



colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
for i in range(len(colors)):
     x = reduced_data_rpca[:, 0][digits.target == i]
     y = reduced_data_rpca[:, 1][digits.target == i]

     plt.scatter(x, y, c=colors[i])
plt.legend(digits.target_names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title("PCA Scatter Plot")
plt.show()




# Import
from sklearn.preprocessing import scale

# Apply `scale()` to the `digits` data
data = scale(digits.data)

from sklearn.cross_validation import train_test_split

# Split the data into training and test sets
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(digits.data, digits.target, digits.images, test_size=0.25, random_state=42)

# Number of training features
n_samples, n_features = X_train.shape
n_samples
n_features

images_train.shape

n_digits = len(np.unique(y_train))
n_digits
y_train.shape[0]
y_train[0]

y_train = y_train.reshape(len(y_train),1)
y_test= y_test.reshape(len(y_test),1)

y_train.shape
y_train
from sklearn.preprocessing import OneHotEncoder

onehot_encoder = OneHotEncoder(sparse=False)
y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)
y_test
# HPELM neural network model
clf = hp.ELM(in_size,10, classification="c" , batch=1000)
clf.add_neurons(10, 'sigm')
clf.add_neurons(10,'lin')
print(str(clf))
clf.train(X_train,y_train)

predicted_array = clf.predict(X_train)
clf.error(predicted_array, y_train)

test_ev= clf.predict(X_test)

print(clf.error(test_ev,y_test))
