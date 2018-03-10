import time

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from hvass_utils import cifar10

import hpelm
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import numpy as np
import itertools




### dataset path goes here
data_path = parentdir + '/cifar10_data/'

cifar10.maybe_download_and_extract(data_path)

# train data
train_data, train_cls, train_labels = cifar10.load_training_data(data_path)

# load test data
test_data, cls_test, test_labels = cifar10.load_test_data(data_path)

print("Size of:")
print("- Training-set:\t\t{}".format(len(train_data)))
print("- Test-set:\t\t{}".format(len(test_data)))


def plain_ELM(name, training_instances, training_labels, test_instances, test_labels,
              predict_instances,
              hidden_layer_size, precision='single',
              neuron_type= 'sigm'):


    model = hpelm.ELM(training_instances.shape[1], training_labels.shape[1], accelerator='GPU', classification='c',
                      batch=2000, precision=precision)

    model.add_neurons(hidden_layer_size, neuron_type)

    print("MODEL: %s" % name)
    print(str(model)) # print model

    model.train(training_instances, training_labels, 'c')

    y_train_predicted = model.predict(training_instances)

    # training accuracy
    y_train_sk = training_labels.argmax(1)
    y_train_predicted_sk = y_train_predicted.argmax(1)
    acc_score_train = accuracy_score(y_train_sk, y_train_predicted_sk)
    print("Train Accuracy: %.5f" % acc_score_train)

    y_test_predicted = model.predict(test_instances)

    # test accuracy
    y_test_sk = test_labels.argmax(1)
    y_test_predicted_sk = y_test_predicted.argmax(1)
    acc_score_test = accuracy_score(y_test_sk, y_test_predicted_sk)
    print("Test Accuracy: %.5f" % acc_score_test)



    # predictions for subsequent training layers instances
    next_layer_pred = model.predict(predict_instances)



    return y_train_predicted, y_test_predicted, y_test_predicted, next_layer_pred, acc_score_train,acc_score_test



X_train = train_data.astype('float32')
Y_train = train_labels.astype('float32')
X_test = test_data.astype('float32')
y_test = test_labels.astype('float32')

# reshape data
X_train = X_train.reshape((len(X_train), X_train.shape[1] * X_train.shape[2] * X_train.shape[3]))
X_test = X_test.reshape((len(X_test), X_test.shape[1] * X_test.shape[2] * X_test.shape[3]))


print('DATASET')
print('X_train shape ', X_train.shape)
print('y_train shape ', Y_train.shape)
print('X_test shape ', X_test.shape)
print('y_test shape ', y_test.shape)
out_class = len(np.unique(y_test))
print('Num Classes: ', out_class)

scaler = StandardScaler()
x_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


###### HYPERPARAMETERS ##########
hyperpar_dict = {'n_layers' : (2,), # aggregator excluded
                 'n_estimators' : ([50,20],), # aggregator excluded
                 'n_neurons' : ([8196,2048,1024],), # aggregator included
                 'precision' : ('single',),

                 'l_train_size': ([30000,10000,10000],),
                 'neuron_type' : ( 'random',)
                 }


test_set = (X_test, y_test) # test set tuple


run_var = 0 # counting the number of runs


def stacking_ensemble(hyperpar):


    global run_var

    l_train_size = hyperpar['l_train_size']

    if sum(l_train_size) > x_train.shape[0]:
        raise ValueError("l_train size list sum should be equal to the training set instances")

    print('Starting training for RUN %d' % run_var)
    tstart = time.time()



    for stacking_layer in range(hyperpar['n_layers']):



        if stacking_layer == 0:
            #initialize
            x_train_layer = X_train[:l_train_size[stacking_layer]]
            y_train_layer = Y_train[:l_train_size[stacking_layer]]
            predict_instances = X_train[l_train_size[stacking_layer]:]
            predict_labels = Y_train[l_train_size[stacking_layer]:]
            x_test=X_test
        else:
            x_train_layer = next_layer_pred[:l_train_size[stacking_layer]]
            y_train_layer = predict_labels[:l_train_size[stacking_layer]]
            predict_instances = next_layer_pred[l_train_size[stacking_layer]:] # move on
            predict_labels = predict_labels[l_train_size[stacking_layer]:]
            x_test=X_test_next





        for predictor in range(hyperpar['n_estimators'][stacking_layer]):
            name = 'Estimator n:%d in layer:%d' % (predictor, stacking_layer)

            # account for possible random neuron type per predictor

            if hyperpar['neuron_type'] is 'random':
                n_type = ('sigm', 'tanh')
                rchoice = np.random.random_integers(high=1,low=0)
                neuron = n_type[rchoice]
            else:
                neuron = hyperpar['neuron_type']


            # random neuron number
            neuron_number = int(hyperpar['n_neurons'][stacking_layer])

            print('#'*100)

            [y_train_predicted, y_test_predicted, y_test_predicted, next_layer_predicted,
            acc_score_train, acc_score_test]= plain_ELM(name, x_train_layer,y_train_layer,
                                                             x_test, y_test,
                                                             predict_instances,
                                                             hidden_layer_size=neuron_number,
                                                             precision=hyperpar['precision'],
                                                             neuron_type= neuron,
                                                             )
            if predictor == 0:
                X_test_next = y_test_predicted
                next_layer_pred=next_layer_predicted



            else:
                    # concatenate
                X_test_next = np.concatenate((X_test_next, y_test_predicted), axis=1)


                next_layer_pred = np.concatenate((next_layer_pred, next_layer_predicted), axis=1)










    # aggregator


    name='Aggregator'
    n_layers = hyperpar['n_layers']
    neuron_number=int(hyperpar['n_neurons'][n_layers])

    x_train_layer = next_layer_pred[:l_train_size[n_layers]]

    y_train_layer = predict_labels[:l_train_size[n_layers]]

    [y_train_predicted, y_test_predicted, y_test_predicted, next_layer_predicted,
                 acc_score_train, acc_score_test] = plain_ELM(name, x_train_layer,
                                                              y_train_layer, X_test_next, y_test, X_test_next,
                                                              neuron_number,
                                                              precision=hyperpar['precision'],
                                                              neuron_type='sigm')




    print("RUN %d ended training" % run_var)
    print("Hyperparameters: ")
    print(hyperpar)
    print("Training done in ", (time.time() - tstart), "seconds!!")
    print("###############################################################################################")
    print('Aggregator Train Accuracy Score: %.5f' % acc_score_train)
    print('Aggregator Test Accuracy Score: %.5f' % acc_score_test)

    ## performance summary








    '''
    for i in range(hyperpar['n_layers']):

        if i==0:
            ensemble_dataset.append(train_test_split(X_train, y_train, test_size=0.5))
        else:
            ensemble_dataset.append(train_test_split(ensemble_dataset[0], ensemble_dataset[2], test_size=0.5)) #train
            ensemble_dataset.append(train_test_split(ensemble_dataset[1], ensemble_dataset[3], test_size=0.5)) #test
            
    '''










keys, values = zip(*hyperpar_dict.items())
for v in itertools.product(*values):
            comb = dict(zip(keys, v))

            print("Run: %d" % (run_var)) # Run number
            print("Training with hyperparameter set:")
            print(comb) # current combination of hyperparameters

            stacking_ensemble(comb)


            run_var +=1









