from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, AlphaDropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.initializers import lecun_normal

from keras import regularizers


class mvgg16(object):
    def __init__(self, input_shape, output_shape, weight_decay=0.0005):
        self.weight_decay = weight_decay

        self.input_shape = input_shape
        self.output_shape = output_shape

    def load_params(self):
        self.model.load_weights('cifar10vgg.h5')

    def build_model(self):
        'Build method for modified VGG16 network'

        model = Sequential()
        weight_decay = self.weight_decay


        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.input_shape, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.325))

        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.425))

        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.425))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.425))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.425))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.425))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.425))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.425))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.525))

        model.add(Flatten())

        model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.525))
        model.add(Dense(self.output_shape))
        model.add(Activation('softmax'))



        '''

        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=self.input_shape, kernel_regularizer=regularizers.l2(weight_decay),
                         kernel_initializer=lecun_normal(), bias_initializer='zeros', activation='selu'))
        # model.add(Activation('selu'))
        # model.add(BatchNormalization())
        model.add(AlphaDropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.input_shape, kernel_regularizer=regularizers.l2(weight_decay),
                         kernel_initializer=lecun_normal(), bias_initializer='zeros', activation='selu'))
        # model.add(Activation('selu'))
        # model.add(BatchNormalization())
        model.add(AlphaDropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                         kernel_initializer=lecun_normal(), bias_initializer='zeros', activation='selu'))
        # model.add(Activation('selu'))
        # model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                         kernel_initializer=lecun_normal(), bias_initializer='zeros', activation='selu'))
        # model.add(Activation('selu'))
        # model.add(BatchNormalization())
        model.add(AlphaDropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                         kernel_initializer=lecun_normal(), bias_initializer='zeros', activation='selu'))
        # model.add(Activation('selu'))
        # model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                         kernel_initializer=lecun_normal(), bias_initializer='zeros', activation='selu'))
        # model.add(Activation('selu'))
        # model.add(BatchNormalization())
        model.add(AlphaDropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                         kernel_initializer=lecun_normal(), bias_initializer='zeros', activation='selu'))
        # model.add(Activation('selu'))
        # model.add(BatchNormalization())
        model.add(AlphaDropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                         kernel_initializer=lecun_normal(), bias_initializer='zeros', activation='selu'))
        # model.add(Activation('selu'))
        # model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                         kernel_initializer=lecun_normal(), bias_initializer='zeros', activation='selu'))
        # model.add(Activation('selu'))
        # model.add(BatchNormalization())
        model.add(AlphaDropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                         kernel_initializer=lecun_normal(), bias_initializer='zeros', activation='selu'))
        # model.add(Activation('selu'))
        # model.add(BatchNormalization())
        model.add(AlphaDropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                         kernel_initializer=lecun_normal(), activation='selu'))
        # model.add(Activation('selu'))
        # model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                         kernel_initializer=lecun_normal(), bias_initializer='zeros', activation='selu'))
        # model.add(Activation('selu'))
        # model.add(BatchNormalization())
        model.add(AlphaDropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                         kernel_initializer=lecun_normal(), bias_initializer='zeros', activation='selu'))
        # model.add(Activation('selu'))
        # model.add(BatchNormalization())
        model.add(AlphaDropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                         kernel_initializer=lecun_normal(), bias_initializer='zeros', activation='selu'))
        # model.add(Activation('selu'))
        # model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(AlphaDropout(0.5))

        model.add(Flatten())
        model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=lecun_normal(),
                        bias_initializer='zeros', activation='selu'))
        # model.add(Activation('selu'))
        # model.add(BatchNormalization())

        model.add(AlphaDropout(0.5))
        model.add(Dense(self.output_shape, kernel_initializer=lecun_normal()))
        model.add(Activation('softmax'))
        
        '''

        return model
