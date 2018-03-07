import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from hvass_utils import cifar10


import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, InputLayer, Flatten
from keras.callbacks import TensorBoard
import itertools
import numpy as np

import tensorflow as tf
from lr_scheduler import lr_sgdr

run_var = 0

log_dir = os.getcwd() + '/mlp_vs_elm_cifar10'
data_path = parentdir + '/cifar10_data/'

batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = False

##### HYPERPARAMS ##############

hyperpar = {'mlp': [False,True],  'drop' : [0 , 0.25] }

cifar10.maybe_download_and_extract(data_path)

# train data
images_train, cls_train, labels_train = cifar10.load_training_data(data_path)

# load test data
images_test, cls_test, labels_test = cifar10.load_test_data(data_path)

print("Size of:")
print("- Training-set:\t\t{}".format(len(images_train)))
print("- Test-set:\t\t{}".format(len(images_test)))


# scale input data
x_train = images_train.astype('float32')
x_test = images_test.astype('float32')
#x_train /= 255
#x_test /= 255



def createntrain(hyperpar,save_dir):
    mlp = hyperpar['mlp']
    dropout = hyperpar['drop']


    model = Sequential()
    model.add(InputLayer(input_shape=(32, 32, 3)))
    model.add(Flatten())
    model.add(Dense(100, trainable=mlp))
    model.add(Activation('tanh'))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    print(model.summary())

    opt = keras.optimizers.nadam(lr=0.001,schedule_decay=0.006)

    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])



    ckpt = keras.callbacks.ModelCheckpoint(save_dir + '/model_run:%s_checkpoint.ckpt' % run_var, monitor='val_loss',
                                           verbose=2,
                                           save_best_only=True, save_weights_only=False, mode='auto', period=1)

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
    tensorboard = TensorBoard(log_dir=save_dir, histogram_freq=1,
                              write_graph=True, write_images=False)

    lr_sched = lr_sgdr(eta_min = 0.00001, eta_max= 0.001 , Ti=20*(len(x_train)//batch_size) )

    model.fit(x_train, labels_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, labels_test),
            shuffle=True, callbacks=[tensorboard, ckpt,early_stopping,lr_sched])


    return model


def main():

    global run_var

    # generate every possible combination of hyperparameters
    keys, values = zip(*hyperpar.items())

    for v in itertools.product(*values):
        comb = dict(zip(keys, v))

        print("Run: %d" % (run_var))  # Run number
        print("Training with hyperparameter set:")

        print(comb)

        save_dir = log_dir + '/%s' %str(list(comb.items()))

        model = createntrain(comb,save_dir)
        del model
        keras.backend.clear_session() # required otherwise keras will raise a weird error

        print("RUN number %d COMPLETED\n\n" % run_var)
        print('#'*60)
    print('tensorboard --logdir=%s' % log_dir)





if __name__ == '__main__':
        main()





### BONUS #### DEEP MLP

save_dir = log_dir + '/deep_mlp'

########## DEEP MLP MODEL
mlp = Sequential()
mlp.add(InputLayer(input_shape=(32,32,3)))
mlp.add(Flatten())
mlp.add(Dense(1000))
mlp.add(Activation('relu'))
mlp.add(Dropout(0.25))
mlp.add(Dense(1000))
mlp.add(Activation('relu'))
mlp.add(Dropout(0.5))
# maybe dropout here

mlp.add(Dense(num_classes))
mlp.add(Activation('softmax'))

print(mlp.summary())

opt = keras.optimizers.nadam(lr=0.001, schedule_decay=0.006)

mlp.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])




ckpt= keras.callbacks.ModelCheckpoint(save_dir + '/mlp_checkpoint.ckpt', monitor='val_loss', verbose=0,
                                      save_best_only=True, save_weights_only=False, mode='auto', period=1)

tensorboard = TensorBoard(log_dir=save_dir + '/deep' , histogram_freq=1,
                          write_graph=True, write_images=False)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')

mlp.fit(x_train, labels_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, labels_test),
              shuffle=True, callbacks=[tensorboard,ckpt,early_stopping])


print(mlp.summary())

# Score trained model.
scores = mlp.evaluate(x_test, labels_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

del mlp # purge model from memory


print('tensorboard --logdir=%s' % log_dir)