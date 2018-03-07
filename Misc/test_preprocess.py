import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from CIFAR10.cifar10dataset import train_data, train_labels, test_data, test_labels
from sklearn.preprocessing import OneHotEncoder

'''
input_path= "F:\\Documenti 2\\University\\Magistrale\\Progettazione Sistemi Embedded\\Progetto EMBEDDED\\dog.jpg"
output_path = 'F:\\Documenti 2\\University\\Magistrale\\Progettazione Sistemi Embedded\\Progetto EMBEDDED\\dog_random{}.jpg'
count = 10

gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# load image to array
image = img_to_array(load_img(input_path))

# reshape to array rank 4
image = image.reshape((1,) + image.shape)

# let's create infinite flow of images
images_flow = gen.flow(image, batch_size=1)
for i, new_images in enumerate(images_flow):
    # we access only first image because of batch_size=1
    new_image = array_to_img(new_images[0], scale=True)
    new_image.save(output_path.format(i + 1))
    if i >= count:
        break
'''

X_train = train_data[:30]
y_train = train_labels[:30]
y_train = y_train.reshape(-1, 1)
onehot_encoder = OneHotEncoder(sparse=False)
y_train = onehot_encoder.fit_transform(y_train)

gen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    fill_mode='nearest',
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
gen.fit(X_train)

i = 0
for X_batch, y_batch in gen.flow(X_train, y_train, batch_size=25):
    if i == 0:
        X = X_batch
        y = y_batch
    else:
        X = np.append(X, X_batch, axis=0)
        y = np.append(y, y_batch, axis=0)
    i += 1
    if i >= 10:
        break

import matplotlib.pyplot as plt

for i in range(10):
    plt.subplot(7, 10, i + 1)
    plt.imshow(X_train[i].astype('uint8'))
    plt.axis('off')
    if i == 0:
        plt.title('Original Image')

for i in range(10, 70):
    plt.subplot(7, 10, i + 1)
    plt.imshow(X[i].astype('uint8'))
    plt.axis('off')
    if i == 10:
        plt.title('Processed Image')
plt.show()
