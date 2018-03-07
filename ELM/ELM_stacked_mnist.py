from Misc.loadMNIST_orig import X_test, y_test, X_train, y_train  # already scaled
import time
import hpelm
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


def plain_ELM(name, training_instances, training_labels, test_instances, test_labels,
              hidden_layer_size):  # TODO more hyperpar

    model = hpelm.ELM(training_instances.shape[1], training_labels.shape[1], accelerator='GPU', classification='c',
                      batch=hidden_layer_size, precision='single')
    model.add_neurons(hidden_layer_size, 'sigm')
    #    model.add_neurons(training_instances.shape[1], 'lin')
    print('\n' + name)
    print(str(model))
    print('X_train shape ', training_instances.shape)
    print('y_train shape ', training_labels.shape)
    print('X_test shape ', test_instances.shape)
    print('y_test shape ', test_labels.shape)

    t = time.time()
    model.train(training_instances, training_labels, 'c')
    elapsed_time_train = time.time() - t

    y_train_predicted = model.predict(training_instances)
    y_test_predicted = model.predict(test_instances)

    print(name + "_Training time: %f" % elapsed_time_train)
    print(name + '_Training Accuracy: ', (1 - model.error(training_labels, y_train_predicted)))
    print(name + '_Test Accuracy: %f \n' % (1 - model.error(test_labels, y_test_predicted)))

    return y_train_predicted, y_test_predicted


np.random.seed(42)

print('DATASET SHAPE')
print('X_train shape ', X_train.shape)
print('y_train shape ', y_train.shape)
print('X_test shape ', X_test.shape)
print('y_test shape ', y_test.shape)
out_class = len(np.unique(y_test))
print('Num Classes: ', out_class)

# Reshape and apply OneHotEncoder to compute the 10class classifier
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
onehot_encoder = OneHotEncoder(sparse=False)
y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)

# Hyperparameters
hidden_layer_size = 1024
n_pred = 3
n_layers = 2

stack_train_inputs = [X_train, y_train]
stack_test_inputs = [X_test, y_test]

for stacking_layer in range(n_layers):
    for predictor in range(n_pred):
        name = 'ELM_layer:%d_n:%d' % (stacking_layer, predictor)
        [y_train_pred, y_test_pred] = plain_ELM(name,
                                                stack_train_inputs[0],
                                                stack_train_inputs[1],
                                                stack_test_inputs[0],
                                                stack_test_inputs[1],
                                                hidden_layer_size
                                                )
        if predictor == 0:
            stack_train_inputs_layer = np.array(y_train_pred)
            stack_test_inputs_layer = np.array(y_test_pred)
        else:
            stack_train_inputs_layer = np.concatenate((stack_train_inputs_layer, y_train_pred), axis=1)
            stack_test_inputs_layer = np.concatenate((stack_test_inputs_layer, y_test_pred), axis=1)

    stack_train_inputs[0] = stack_train_inputs_layer
    stack_test_inputs[0] = stack_test_inputs_layer
    # Scaling data (mean 0, variance 1)
    scaler = StandardScaler()
    stack_train_inputs[0] = scaler.fit_transform(stack_train_inputs[0])
    stack_test_inputs[0] = scaler.transform(stack_test_inputs[0])

[y_train_pred, y_test_pred] = plain_ELM('AGGREGATOR',
                                        stack_train_inputs[0],
                                        stack_train_inputs[1],
                                        stack_test_inputs[0],
                                        stack_test_inputs[1],
                                        hidden_layer_size
                                        )
