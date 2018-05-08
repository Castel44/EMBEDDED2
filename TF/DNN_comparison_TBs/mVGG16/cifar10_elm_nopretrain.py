import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
import pickle
import numpy as np
from cyclic_lr import CyclicLR
from skimage.color import rgb2ycbcr, rgb2yuv

from vgg16.vgg16 import mvgg16

import os

####################################
#########################
##############

run = "run_%d" % 6

log_dir = os.path.join("./tensorboard", run)

load_params= False


############################
###################
##############

def normalize(X_train, X_test):


    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)

    return X_train, X_test

train, test = cifar10.load_data()
x_train, y_train = train
x_test, y_test = test
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)
x_train = rgb2yuv(x_train)
x_test = rgb2yuv(x_test)

x_train, x_test = normalize(x_train,x_test)




batch_size = 128
n_epochs = 500
learning_rate = 0.12
zca_epsilon = 0.01 #0.01
lr_decay = 1e-6


input_size = [32, 32, 3]
output_size = 10


vgg = mvgg16(input_size, output_size)
model = vgg.build_model()

if load_params is True: model.load_weights(log_dir +'/cifar10vgg.h5')

datagen_train = ImageDataGenerator(
    rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen_train.fit(x_train)


datagen_test = ImageDataGenerator(
)

datagen_test.fit(x_train)


opt = keras.optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)  #keras.optimizers.nadam(lr=learning_rate, schedule_decay=lr_decay)

tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,
                                          write_images=False)



model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])



#n_epochs = 10
#n_iterations =  n_epochs*(x_train.shape[0] // batch_size)

#clr_triangular = CyclicLR(mode='triangular',step_size=n_iterations, base_lr=1e-7, max_lr=1e-6, plotting=True)
#clr = CyclicLR(base_lr=1e-7, max_lr=1e-5, step_size=int((len(x_train)//batch_size)*2), mode='triangular2')

lr_drop = 20

def lr_scheduler(epoch):
    return learning_rate * (0.5 ** (epoch // lr_drop))


reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

ckpt = keras.callbacks.ModelCheckpoint(log_dir +'/checkpoint.ckpt', monitor='val_loss',
                                           verbose=2,
                                           save_best_only=False, save_weights_only=True, mode='auto', period=2)

model.load_weights('cifar10vgg.h5')

model2= keras.Model(inputs=model.input, outputs=model.layers[-7].output)
x_train = model2.predict(x_train,batch_size=128)
x_test = model2.predict(x_test,batch_size=128)


from tfelm.elm import ELM
import tensorflow as tf

ml_elm1 = ELM(512, output_size, l2norm=10e3)
ml_elm1.add_layer(4000)
ml_elm1.compile()

ml_elm1.fit(x_train, y_train)

ml_elm1.evaluate(x_test, y_test)










