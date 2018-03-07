from Misc.loadMNIST_orig import X_test, y_test, X_train, y_train  # already scaled
import time
import hpelm
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def plain_ELM(name, training_instances, training_labels, test_instances, test_labels, hidden_layer_size,
              training_next_layer=None):  # TODO more hyperpar

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

    if training_next_layer is None:
        return y_train_predicted, y_test_predicted
    else:
        y_train_predicted = model.predict(training_next_layer)
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
hidden_layer_size = 1000
n_pred = 3
n_layers = 2

# SHIT: test work for 3 layer of n_pred

# Split dataset in n_layer subset, each layer is trained with different datasets
X_train1, X_train2, y_train1, y_train2 = train_test_split(X_train, y_train, test_size=0.5, shuffle=False)

# stack_train_inputs = [X_train1, y_train1, X_train2, y_train2]
# stack_test_inputs = [X_test, y_test]

# 1st layer
for predictor in range(n_pred):
    name = 'ELM_layer:%d_n:%d' % (0, predictor)
    [y_train_pred, y_test_pred] = plain_ELM(name,
                                            X_train1,
                                            y_train1,
                                            X_test,
                                            y_test,
                                            hidden_layer_size,
                                            training_next_layer=X_train2
                                            )
    if predictor == 0:
        train_next_layer = np.array(y_train_pred)
        test_next_layer = np.array(y_test_pred)
    else:
        train_next_layer = np.concatenate((train_next_layer, y_train_pred), axis=1)
        test_next_layer = np.concatenate((test_next_layer, y_test_pred), axis=1)

# Scaling data (mean 0, variance 1)
scaler = StandardScaler()
train_next_layer = scaler.fit_transform(train_next_layer)
test_next_layer = scaler.transform(test_next_layer)

# 2nd layer
for predictor in range(n_pred):
    name = 'ELM_layer:%d_n:%d' % (1, predictor)
    [y_train_pred, y_test_pred] = plain_ELM(name,
                                            train_next_layer,
                                            y_train2,
                                            test_next_layer,
                                            y_test,
                                            hidden_layer_size
                                            )
    if predictor == 0:
        train_blender = np.array(y_train_pred)
        test_blender = np.array(y_test_pred)
    else:
        train_blender = np.concatenate((train_blender, y_train_pred), axis=1)
        test_blender = np.concatenate((test_blender, y_test_pred), axis=1)

# Scaling data (mean 0, variance 1)
scaler = StandardScaler()
train_blender = scaler.fit_transform(train_blender)
test_blender = scaler.transform(test_blender)

# Final Blender
[y_train_pred_L1, y_test_pred_L1] = plain_ELM('AGGREGATOR',
                                              train_blender,
                                              y_train2,
                                              test_blender,
                                              y_test,
                                              hidden_layer_size
                                              )
