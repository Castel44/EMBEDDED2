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
              predict_instances, predict_labels,
              agg_x_train, agg_y_train,# these are necessary
              hidden_layer_size, precision='single',
              neuron_type= 'sigm',training_type= None):


    model = hpelm.ELM(training_instances.shape[1], training_labels.shape[1], accelerator='GPU', classification='c',
                      batch=2000, precision=precision)

    model.add_neurons(hidden_layer_size, neuron_type)

    print("MODEL: %s" % name)
    print(str(model)) # print model

    model.train(training_instances, training_labels, 'c', training_type)

    y_train_predicted = model.predict(training_instances)

    # training accuracy
    y_train_sk = training_labels.argmax(1)
    y_train_predicted_sk = y_train_predicted.argmax(1)
    acc_score_train = accuracy_score(y_train_sk, y_train_predicted_sk)

    y_test_predicted = model.predict(test_instances)

    # test accuracy
    y_test_sk = test_labels.argmax(1)
    y_test_predicted_sk = y_test_predicted.argmax(1)
    acc_score_test = accuracy_score(y_test_sk, y_test_predicted_sk)

    if name is not 'aggregator':
        # predictions for subsequent training layers instances
        next_layer_pred = model.predict(predict_instances)
        y_pred = predict_labels.argmax(1)
        acc_score_validation = accuracy_score(y_pred, next_layer_pred)

        # predictions for training the aggregator
        agg_pred = model.predict(agg_x_train)

    else:
        next_layer_pred=0
        agg_pred=0
        acc_score_validation= 0


    return y_train_predicted, y_test_predicted, next_layer_pred, agg_pred, acc_score_train,acc_score_test, acc_score_validation



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
hyperpar_dict = {'n_layers' : (4,),
                 'n_estimators' : (5,10,50),
                 'n_neurons' : (1000,2000,'random'),
                 'precision' : ('single', 'double'),
                 'scaling' : (False,True),
                 'training_type' : (None, 'OP'),
                 'agg_train_size': (0.2,),
                 'neuron_type' : ('sigm', 'random')
                 }




test_set = (X_test,y_test) # test set tuple



run_var = 0 # counting the number of runs


def stacking_ensemble(hyperpar):

    global run_var

    # split train set into n_layers sets
    #ensemble_dataset = []

    #splits = KFold(n_splits=hyperpar['n_layers'])
    #print("Number of splits: %d" % splits.get_n_splits(X_train,y_train))
    #splits.split(X_train.shape[0])
    agg_size = int(hyperpar['agg_train_size']*x_train.shape[0])
    t_size = int(x_train.shape[0]* (1-hyperpar['agg_train_size']))
    # split the aggregator train set and actual train sets for layers
    X_train = x_train[:t_size]
    y_train = Y_train[:t_size]

    agg_x_train=x_train[agg_size:]
    agg_y_train = Y_train[agg_size:]


    # splitting the train set
    split_indx = X_train.shape[0] // hyperpar['n_layers']



    #stack_train_inputs = [X_train, y_train]
    #stack_test_inputs = [X_test, y_test]


    # start training and testing the ensemble
    print('Starting training for RUN %d' % run_var)
    tstart = time.time()

    #next_layer_pred = 0


    for stacking_layer in range(hyperpar['n_layers']-1):



        if stacking_layer == 0:
            x_train_layer = X_train[split_indx * (stacking_layer):split_indx * (stacking_layer + 1)]
            y_train_layer = y_train[split_indx * (stacking_layer):split_indx * (stacking_layer + 1)]
            predict_instances = X_train[split_indx * (stacking_layer + 1):]
            predict_labels = y_train[split_indx * (stacking_layer + 1):]
        else:
            x_train_layer = next_layer_pred #maybe not need to reference
            y_train_layer = predict_labels # from previopus layer
            predict_instances = X_train[split_indx * (stacking_layer + 1):] # move on
            predict_labels = y_train[split_indx * (stacking_layer + 1):]




        for predictor in range(hyperpar['n_estimators']):
            name = 'Estimator n:%d in layer:%d' % (predictor, stacking_layer)

            # account for possible random neuron type per predictor

            if hyperpar['neuron_type'] is 'random':
                n_type = ('sigm', 'tanh')
                rchoice = np.random.random_integers(0,1)
                neuron = n_type[rchoice]
            else:
                neuron = hyperpar['neuron_type']


            # random neuron number
            if hyperpar['n_neurons'] is 'random':  # scale linearly n neurons for each layer

                neuron_number = np.random.random_integers(1000/(1+predictor),2000/(1+predictor))

            else:
                neuron_number = int(hyperpar['n_neurons']/(1+predictor))

                print('#'*100)

                [y_train_predicted, y_test_predicted, next_layer_pred, agg_pred,
                 acc_score_train, acc_score_test, acc_score_validation]= plain_ELM(name, x_train_layer,
                                                                                   y_train_layer, X_test, y_test,
                                                                                   predict_instances,predict_labels,
                                                                                   agg_x_train,agg_y_train,
                                                                                   neuron_number,precision=hyperpar['precision'],
                                                                                   neuron_type= neuron,
                                                                                   training_type= hyperpar['training_type'])







    # aggregator


    name='Aggregator'

    [y_train_predicted, y_test_predicted, next_layer_pred, agg_pred,
     acc_score_train, acc_score_test, acc_score_validation] = plain_ELM(name, agg_x_train,
                                                                        agg_y_train, X_test, y_test,
                                                                        predict_instances= X_test, predict_labels= y_test,
                                                                        agg_x_train=X_test,# not very efficient but works
                                                                        hidden_layer_size=500,
                                                                        precision=hyperpar['precision'],
                                                                        neuron_type='sigm',
                                                                        training_type=hyperpar['training_type'])



    print("RUN %d ended training" % run_var)
    print("Hyperparameters: ")
    print(hyperpar)
    print("Training done in ", (time.time() - tstart), "seconds!!")
    print("###############################################################################################")

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









