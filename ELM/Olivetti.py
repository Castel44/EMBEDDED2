from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import hpelm
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def plot_images(instances, images_per_row=20, size=64, **options):
    plt.figure()
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row: (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap='binary', **options)
    plt.axis("off")
    plt.show()


olivetti = datasets.fetch_olivetti_faces()
x_train, x_test, y_train, y_test = train_test_split(olivetti.data.astype('float32'), olivetti.target.astype('float32'), shuffle=True,
                                                    test_size=0.2)

print('Olivetti faces dataset')
print('x_train shape ', x_train.shape)
print('y_train shape ', y_train.shape)
print('x_test shape ', x_test.shape)
print('y_test shape ', y_test.shape)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
onehot_encoder = OneHotEncoder(sparse=False)
y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.transform(y_test)


#HYPERPARAMETERS
neuron_number = 200
out_class = 40

print('ELM-GPU single')
elm = hpelm.ELM(x_train.shape[1], out_class, classification="c", batch=1000, accelerator="GPU",
                precision='single', norm=10**0)
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


