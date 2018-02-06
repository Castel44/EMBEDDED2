import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


digits_tra = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra", header=None)
print(digits_tra.shape)
digits_tes = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes", header=None)
print(digits_tes.shape)

x_train = digits_tra.values[:,0:64]
y_train = digits_tra.values[:,64]
print(x_train.shape,y_train.shape)
x_test = digits_tes.values[:,0:64]
y_test = digits_tes.values[:,64]
print(x_test.shape,y_test.shape)

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

# Let's create a plot_digits() function that will draw a scatterplot, with little image of digit
from sklearn.preprocessing import MinMaxScaler
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib

def plot_digits(X, y, min_distance=0.05, images=None, figsize=(13, 10)):
    # Let's scale the input features so that they range from 0 to 1
    X_normalized = MinMaxScaler().fit_transform(X)
    # Now we create the list of coordinates of the digits plotted so far.
    # We pretend that one is already plotted far away at the start, to
    # avoid `if` statements in the loop below
    neighbors = np.array([[10., 10.]])
    # The rest should be self-explanatory
    plt.figure(figsize=figsize)
    cmap = matplotlib.cm.get_cmap("jet")
    digits = np.unique(y)
    for digit in digits:
        plt.scatter(X_normalized[y == digit, 0], X_normalized[y == digit, 1], c=cmap(digit / 9))
    plt.axis("off")
    ax = plt.gcf().gca()  # get current axes in current figure
    for index, image_coord in enumerate(X_normalized):
        closest_distance = np.linalg.norm(np.array(neighbors) - image_coord, axis=1).min()
        if closest_distance > min_distance:
            neighbors = np.r_[neighbors, [image_coord]]
            if images is None:
                plt.text(image_coord[0], image_coord[1], str(int(y[index])),
                         color=cmap(y[index] / 9), fontdict={"weight": "bold", "size": 16})
            else:
                image = images[index].reshape(8, 8) #8x8 digit image
                imagebox = AnnotationBbox(OffsetImage(image, cmap="binary"), image_coord)
                ax.add_artist(imagebox)
    plt.show()




# PCA for visualize data, not great, but blazing fast
from sklearn.decomposition import PCA
# Create a Randomized PCA model that takes two components
randomized_pca = PCA(n_components=2, random_state=42)
# Fit and transform the data to the model reduced_data_rpca = randomized_pca.fit_transform(digits.data)
# Create a regular PCA model pca = PCA(n_components=2)
# Fit and transform the data to the model
reduced_data_rpca = randomized_pca.fit_transform(x_train)
type(reduced_data_rpca)
# Inspect the shape
reduced_data_rpca.shape
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

# plot_digits(reduced_data_rpca, y_train, images=x_train, figsize=(35, 25))

'''
# TSNE + PCA best way to visualize data, but kinda slow
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
pca_tsne = Pipeline([
    ("pca", PCA(n_components=0.95, random_state=42)),
    ("tsne", TSNE(n_components=2, random_state=42)),
])
X_reduced = pca_tsne.fit_transform(x_train)
plt.figure(figsize=(13,10))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_train, cmap="jet")
plt.axis('off')
plt.colorbar()
plt.show()
'''


# Shuffle data
np.random.seed(42)
rnd_idx = np.random.permutation(3823)
x_train = x_train[rnd_idx]
y_train = y_train[rnd_idx]

#Scaling data (mean 0, variance 1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
