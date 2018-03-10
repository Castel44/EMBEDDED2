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
              hidden_layer_size):  # TODO more hyperpar

    model = hpelm.ELM(training_instances.shape[1], training_labels.shape[1], accelerator='GPU', classification='c',
                      batch=1000, precision='single')
    model.add_neurons(hidden_layer_size, 'sigm')
    #    model.add_neurons(training_instances.shape[1], 'lin')
    print(str(model))
    t = time.time()
    model.train(training_instances, training_labels, 'c')
    elapsed_time_train = time.time() - t
    y_train_predicted = model.predict(training_instances)

    print(name + "_Training time: %f" % elapsed_time_train)
    print(name + '_Training Error: ', model.error(training_labels, y_train_predicted))
    y_test_predicted = model.predict(test_instances)
    print(name + '_Test Error: ', model.error(test_labels, y_test_predicted))

    # print(y_test_predicted)

    # print(model.confusion(y_test,y_test_predicted))

    y_train_sk = training_labels.argmax(1)  # one hot
    y_train_predicted_sk = y_train_predicted.argmax(1)
    acc_score_train = accuracy_score(y_train_sk, y_train_predicted_sk)
    print(name + "_Train_Accuracy:\n", acc_score_train)

    y_test_sk = test_labels.argmax(1)  # one hot
    y_test_predicted_sk = y_test_predicted.argmax(1)
    acc_score_test = accuracy_score(y_test_sk, y_test_predicted_sk)
    print(name + "_Test_Accuracy:\n", acc_score_test)
    # cnf_matrix = confusion_matrix(y_test_sk, y_test_predicted_sk)
    # class_report = classification_report(y_test_sk, y_test_predicted_sk)
    # np.set_printoptions(precision=2)
    # print("Accuracy:\n", acc_score)
    # print("Confusion Matrix:\n", cnf_matrix)
    # print("Classification report\n: ", class_report)

    return y_train_predicted, y_test_predicted


X_train = train_data.astype('float32')
y_train = train_labels.astype('float32')
X_test = test_data.astype('float32')
y_test = test_labels.astype('float32')

print('Reshape and scaling data (mean 0, std 1)')
X_train = X_train.reshape((len(X_train), X_train.shape[1] * X_train.shape[2] * X_train.shape[3]))
X_test = X_test.reshape((len(X_test), X_test.shape[1] * X_test.shape[2] * X_test.shape[3]))


print('DATASET')
print('X_train shape ', X_train.shape)
print('y_train shape ', y_train.shape)
print('X_test shape ', X_test.shape)
print('y_test shape ', y_test.shape)
out_class = len(np.unique(y_test))
print('Num Classes: ', out_class)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

'''
print('Reshape and apply OneHotEncoder')
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
onehot_encoder = OneHotEncoder(sparse=False)
y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)
'''

# Hyperparameters
hidden_layer_size = 8192


layers_size = [8192,1024,256]
n_pred = [10,5]
n_layers = 2

stack_train_inputs = [X_train, y_train]
stack_test_inputs = [X_test, y_test]

for stacking_layer in range(n_layers):
    for predictor in range(n_pred[stacking_layer]):
        name = 'ELM_layer:%d_n:%d' % (stacking_layer, predictor)
        [y_train_pred, y_test_pred] = plain_ELM(name,
                                                stack_train_inputs[0],
                                                stack_train_inputs[1],
                                                stack_test_inputs[0],
                                                stack_test_inputs[1],
                                                layers_size[stacking_layer]
                                                )
        if predictor == 0:
            stack_train_inputs_layer = np.array(y_train_pred)
            stack_test_inputs_layer = np.array(y_test_pred)
        else:
            stack_train_inputs_layer = np.concatenate((stack_train_inputs_layer, y_train_pred), axis=1)
            stack_test_inputs_layer = np.concatenate((stack_test_inputs_layer, y_test_pred), axis=1)

    stack_train_inputs[0] = stack_train_inputs_layer
    stack_test_inputs[0] = stack_test_inputs_layer

[y_train_pred, y_test_pred] = plain_ELM('AGGREGATOR',
                                        stack_train_inputs[0],
                                        stack_train_inputs[1],
                                        stack_test_inputs[0],
                                        stack_test_inputs[1],
                                        layers_size[stacking_layer+1]
                                        )
