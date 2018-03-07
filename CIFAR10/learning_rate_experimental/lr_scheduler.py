import keras
from keras import backend as K
import numpy as np

import matplotlib.pyplot as plt # for plotting

# inspired by https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0
# article https://arxiv.org/pdf/1506.01186.pdf Cyclical Learning Rates for Training Neural Networks
# http://teleported.in/posts/cyclic-learning-rate/



class lr_sgdr(keras.callbacks.History):
    ''' SGDR lr schedule Stochastic Gradient descent with Warm restarts.
        from https://arxiv.org/pdf/1608.03983.pdf

    '''


    def __init__(self, eta_min, eta_max, Ti,
                 Tmult=1, eta_decay=1 ): # val_loss for validation

        super(lr_sgdr, self).__init__()

        self.eta_min = eta_min # minimum value for learning rate
        self.eta_max = eta_max # max value for learning rate
        self.Ti = Ti # epoch period for oscillating function
        self.Tmult= Tmult
        self.Tmult = Tmult
        self.eta_decay = eta_decay
        self.Tcur = 0
        self.eta = eta_max


    def on_batch_begin(self, batch, logs=None): ## here


        self.eta = self.eta_min + 0.5*(self.eta_max-self.eta_min)*(1+ np.cos((self.Tcur/self.Ti)*np.pi))
        # check for restart
        if self.Tcur == self.Ti: # restart
            # increase Ti by Tmult
            self.Ti += self.Tmult # multiply or add
            # decrease eta_min and eta_max
            self.eta_min /= self.eta_decay
            self.eta_max /= self.eta_decay

        self.Tcur +=1
        print("updating eta to: %.7f" % self.eta)
        K.set_value(self.model.optimizer.lr, self.eta)

        return K.get_value(self.model.optimizer.lr)


class lr_cyclic(keras.callbacks.History):
    ''' SGDR lr schedule Stochastic Gradient descent with Warm restarts.
        from https://arxiv.org/pdf/1608.03983.pdf

    '''


    def __init__(self, eta_min, eta_max, Ti,
                 Tmult=1, eta_decay=1,): # val_loss for validation

        super(lr_sgdr, self).__init__()

        self.eta_min = eta_min # minimum value for learning rate
        self.eta_max = eta_max # max value for learning rate
        self.Ti = Ti # epoch period for oscillating function
        self.Tmult= Tmult
        self.Tmult = Tmult
        self.eta_decay = eta_decay
        self.Tcur = 0
        self.eta = eta_max


    def on_batch_begin(self, batch, logs=None): ## here

        self.eta = self.eta_min + 0.5*(self.eta_max-self.eta_min)*(1+ np.cos((self.Tcur/self.Ti)*np.pi)) #triangle cyclic
        # check for restart
        if self.Tcur == self.Ti: # restart
            # increase Ti by Tmult
            self.Ti += self.Tmult # multiply or add
            # decrease eta_min and eta_max
            self.eta_min /= self.eta_decay
            self.eta_max /= self.eta_decay

        self.Tcur +=1
        print("updating eta to: %.7f" % self.eta)
        K.set_value(self.model.optimizer.lr, self.eta)

        return K.get_value(self.model.optimizer.lr)


class lr_finder(keras.callbacks.History):

    def __init__(self,eta_min = 0.000001, step=100, loss_type='loss', plotting = True): # val_loss for validation

        super(lr_finder, self).__init__()

        self.loss_type = loss_type
        self.eta = [eta_min]
        self.step = step
        self.loss = [0]
        self.plotting = plotting



    def on_epoch_begin(self, epoch, logs=None):
        
        if epoch > 1:
            _ = self.eta[-1] + self.eta[0] * self.step

            self.eta.append(_)

            K.set_value(self.model.optimizer.lr, self.eta[-1])

            self.loss.append(self.history[self.loss_type][-1])


        else:

            K.set_value(self.model.optimizer.lr, self.eta[0])


        return self.eta, self.loss



    def on_train_end(self, logs=None):

        if self.plotting is True:
            # plt.clf() # clear all figures maybe not needed

            fig = plt.plot(self.eta, self.loss)
            plt.xlabel('learning_rate')
            plt.ylabel(self.loss_type)
            plt.show(fig)


'''
    def on_batch_end(self, batch, logs={}):

        _ = self.eta[-1] + self.eta[0] * self.step

        self.eta.append(_)

        K.set_value(self.model.optimizer.lr, self.eta[-1])

        self.loss.append(logs.get('loss'))

        return self.eta,self.loss
        
'''







'''
    def on_epoch_begin(self, epoch, logs=None):

        if epoch > 0:

            _ = self.eta[-1] + self.eta[0] * self.step

            self.eta.append(_)

            K.set_value(self.model.optimizer.lr, self.eta[-1])

            self.loss.append(self.history[self.loss_type][-1])

            if self.plotting is True:
                # plt.clf() # clear all figures maybe not needed

                fig = plt.plot(self.eta, self.loss)
                plt.xlabel('learning_rate')
                plt.ylabel(self.loss_type)




        return self.eta, self.loss
        
        '''









'''
    def on_epoch_begin(self, epoch, logs=None):

        if epoch == 0:

            return

        else:
            # smoothed_loss = np.inf
            # get current loss:
            current_loss = self.history[self.loss_type][-1]  # self.history[self.loss_type]
            current_lr = K.get_value(self.model.optimizer.lr)



            print("#"*100)

            print("current threshold: %.E5" % self.threshold)

            if epoch == 1:
                # initialize filter

                self.smoothed_loss = current_loss


            else:

                self.smoothed_loss = self.forgetting_factor * self.smoothed_loss + (1 - self.forgetting_factor) * current_loss

                if np.abs(self.smoothed_loss - current_loss) < self.threshold:  # plateau

                    print("Dropping learning rate from %.E5 by a %.3f factor" % (current_lr, self.drop_factor))

                    current_lr /= self.drop_factor

                    K.set_value(self.model.optimizer.lr, current_lr)

                    self.threshold *= 10

                    # threshold decay with epochs simple linear model
                self.threshold = self.threshold - self.threshold_decay * epoch * self.threshold

                # elif (smoothed_loss - current_loss):

            return K.get_value(self.model.optimizer.lr)
'''