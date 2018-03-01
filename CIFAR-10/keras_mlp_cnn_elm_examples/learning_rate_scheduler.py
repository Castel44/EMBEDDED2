import keras
from keras import backend as K
import numpy as np

class lr_scheduler(keras.callbacks.History):

    def __init__(self, initial_lr, forgetting_factor, drop_factor=10, threshold=0.01,
                 threshold_decay=0.1, loss_type='train_loss'):

        super(lr_scheduler, self).__init__()

        self.initial_lr = initial_lr
        self.forgetting_factor = forgetting_factor
        self.drop_factor = drop_factor
        self.threshold = threshold
        self.threshold_decay = threshold_decay
        self.loss_type = loss_type

    def on_epoch_begin(self, epoch, logs=None):

        smoothed_loss = np.infinity
        # get current loss:
        current_loss = self.history[self.loss_type]
        current_lr = K.get_value(self.model.optimizer.lr)

        # threshold decay with epochs simple linear model
        self.threshold = self.threshold - self.threshold_decay * epoch * self.threshold

        print("current threshold: %.4f" % self.threshold)

        if epoch == 0:
            # initialize filter
            smoothed_loss = current_loss

        else:

            smoothed_loss = self.forgetting_factor * smoothed_loss + (1 - self.forgetting_factor) * current_loss

            if (smoothed_loss - current_loss) < self.threshold: # plateau

                print("Dropping learning rate from %.E5 by a %.3f factor" % (current_lr,self.drop_factor))

                current_lr  /= self.drop_factor

                K.set_value(self.model.optimizer.lr, current_lr )

            #elif (smoothed_loss - current_loss):

        return K.get_value(self.model.optimizer.lr)

