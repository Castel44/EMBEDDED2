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
from lr_scheduler import lr_sgdr,lr_finder

from cyclic_lr import CyclicLR

run_var = 0

log_dir = os.getcwd() + '/mlp_vs_elm_cifar10'
data_path = parentdir + '/cifar10_data/'

batch_size = 64
num_classes = 10
epochs = 200
data_augmentation = True

##### HYPERPARAMS ##############
cyclic = keras.optimizers.adam(lr=0.0001)
nadam = keras.optimizers.nadam(lr=0.0001, schedule_decay=0.0001)
sgdr = keras.optimizers.nadam(lr=0.001) #keras.optimizers.SGD(lr=0.001,nesterov=True)
adam = keras.optimizers.adam(lr=0.0001)

hyperpar = {'opt' : [adam]}

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


def createntrain(hyperpar,save_dir):

    model = Sequential()
    model.add(InputLayer(input_shape=(32, 32, 3)))
    model.add(Flatten())
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    print(model.summary())

    opt = hyperpar['opt']

    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])



    ckpt = keras.callbacks.ModelCheckpoint(save_dir + '/model_run:%s_checkpoint.ckpt' % run_var, monitor='val_loss',
                                           verbose=2,
                                           save_best_only=True, save_weights_only=False, mode='auto', period=1)

    #early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
    tensorboard = TensorBoard(log_dir=save_dir, histogram_freq=1,
                              write_graph=True, write_images=False)


    if hyperpar['opt'] == sgdr:

        lr_sched = lr_sgdr(eta_min = 0.0001, eta_max= 0.0018, Ti=5*(len(x_train)//batch_size), Tmult=1, eta_decay=1 )

        model.fit(x_train, labels_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, labels_test),
            shuffle=True, callbacks=[tensorboard, ckpt,lr_sched])

    elif hyperpar['opt'] == cyclic:

        clr_triangular = CyclicLR(mode='triangular',step_size=7000, base_lr=0.0000001, max_lr=0.00015)
        #clr = CyclicLR(base_lr=0.0000001,max_lr=0.0006, step_size=int(50000//batch_size*10))

        model.fit(x_train, labels_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, labels_test),
                  shuffle=True, callbacks=[tensorboard, ckpt, clr_triangular])



    else:



        if data_augmentation is False:


            model.fit(x_train, labels_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=(x_test, labels_test),
                      shuffle=True, callbacks=[tensorboard, ckpt])

        else:

            print('Using real-time data augmentation.')

            clr_triangular = CyclicLR(mode='triangular', step_size=7000, base_lr=0.00000001, max_lr=0.00015)
            #clr = CyclicLR(base_lr=0.00000001, max_lr=0.0006, step_size=int(50000 // batch_size * 10))
            # This will do preprocessing and realtime data augmentation:
            datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=True,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=True,  # divide each input by its std
                zca_whitening=True,  # apply ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images

            # Compute quantities required for feature-wise normalization
            # (std, mean, and principal components if ZCA whitening is applied).
            datagen.fit(x_train)

            model.fit_generator(datagen.flow(x_train, labels_train,
                      batch_size=batch_size),
                      epochs=epochs,
                      validation_data=(x_test, labels_test),
                      workers=4,
                       callbacks=[tensorboard, ckpt, clr_triangular])

            h = clr.history
            lr = h['lr']
            acc = h['acc']
            import matplotlib.pyplot as plt
            plt.plot(lr,acc)
            imp_res_len=100
            imp_res = np.ones(imp_res_len)/imp_res_len
            filtered=np.convolve(acc,imp_res,mode='same')
            plt.plot(lr,filtered)


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


        save_dir = log_dir + '/run:%s' % run_var

        model = createntrain(comb,save_dir)
        del model
        keras.backend.clear_session() # required otherwise keras will raise a weird error


        print("RUN number %d COMPLETED\n\n" % run_var)
        print('#'*60)
    print('tensorboard --logdir=%s' % log_dir)



if __name__ == '__main__':
        main()


'''
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
'''