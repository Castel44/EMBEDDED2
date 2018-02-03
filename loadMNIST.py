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
