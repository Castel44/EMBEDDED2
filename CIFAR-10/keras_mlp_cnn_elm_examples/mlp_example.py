import keras
from hvass_utils import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, InputLayer, Flatten
from keras.callbacks import TensorBoard
import itertools
import os

run_var = 0

data_path = os.getcwd() + '/mlp_vs_elm_cifar10'

batch_size = 32
num_classes = 10
epochs = 50
data_augmentation = False

##### HYPERPARAMS ##############

hyperpar = {'mlp': [False,True],  'drop' : [0,0.25] }

cifar10.maybe_download_and_extract()

# train data
images_train, cls_train, labels_train = cifar10.load_training_data()

# load test data
images_test, cls_test, labels_test = cifar10.load_test_data()

print("Size of:")
print("- Training-set:\t\t{}".format(len(images_train)))
print("- Test-set:\t\t{}".format(len(images_test)))


# scale input data
x_train = images_train.astype('float32')
x_test = images_test.astype('float32')
x_train /= 255
x_test /= 255


def createntrain(hyperpar,save_dir):
    mlp = hyperpar['mlp']
    dropout = hyperpar['drop']

    model = Sequential()
    model.add(InputLayer(input_shape=(32, 32, 3)))
    model.add(Flatten())
    model.add(Dense(1000, trainable=mlp))
    model.add(Activation('tanh'))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    print(model.summary())

    opt = keras.optimizers.nadam(lr=0.001,schedule_decay=0.004)

    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])



    ckpt = keras.callbacks.ModelCheckpoint(save_dir + '/model_run:%s_checkpoint.ckpt' % run_var, monitor='val_loss',
                                           verbose=2,
                                           save_best_only=True, save_weights_only=False, mode='auto', period=1)

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
    tensorboard = TensorBoard(log_dir=save_dir, histogram_freq=1,
                              write_graph=True, write_images=False)

    model.fit(x_train, labels_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, labels_test),
            shuffle=True, callbacks=[tensorboard, ckpt,early_stopping])

    print(model.summary())


def main():

    global run_var

    # generate every possible combination of hyperparameters
    keys, values = zip(*hyperpar.items())

    for v in itertools.product(*values):
        comb = dict(zip(keys, v))

        print("Run: %d" % (run_var))  # Run number
        print("Training with hyperparameter set:")

        print(comb)

        save_dir = data_path + '/%s' %str(list(comb.items()))

        createntrain(comb,save_dir)

        print("RUN number %d COMPLETED\n\n" % run_var)
        print('tensorboard --logdir=%s' % save_dir)





if __name__ == '__main__':
        main()





### BONUS #### DEEP MLP

save_dir = data_path + '/deep_mlp'

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

opt = keras.optimizers.nadam(lr=0.001, schedule_decay=0.004)

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


print('tensorboard --logdir=%s' % data_path)