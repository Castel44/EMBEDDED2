from sklearn import datasets
import time
import hpelm
import numpy as np
from sklearn.metrics import classification_report

iris = datasets.load_iris()
# print(list(iris.keys()))
# print(iris.DESCR)

# Seed for random num gen, high dependence of accuracy from those numbers.
rnd_seed = 42
np.random.seed(10)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state= rnd_seed)

#Scaling data (mean 0, variance 1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Reshape and apply OneHotEncoder to compute the 3class classifier
y_train = y_train.reshape(-1,1)
y_test= y_test.reshape(-1,1)
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False)
y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)


# Building model
print('ELM')
elm = hpelm.ELM(4,3, classification ="c")
elm.add_neurons(100,'sigm')
print(str(elm))
# Training model
t = time.time()
elm.train(x_train,y_train, 'c')
elapsed_time_train = time.time() - t
y_train_predicted = elm.predict(x_train)
print("Training time: %f" % elapsed_time_train)
print('Training Accuracy: ',(1-elm.error(y_train,y_train_predicted)))
# Prediction from trained model
y_test_predicted = elm.predict(x_test)
print('Test Accuracy: ',(1-elm.error(y_test,y_test_predicted)))
print(elm.confusion(y_test,y_test_predicted))
y_test_sk = y_test.argmax(1)
y_test_predicted_sk = y_test_predicted.argmax(1)
class_report = classification_report(y_test_sk, y_test_predicted_sk)
np.set_printoptions(precision=2)
print("Classification report\n: ", class_report)

# LOO ELM
print('P-ELM')
elm = hpelm.ELM(4,3, classification ="c")
elm.add_neurons(100,'sigm')
t = time.time()
elm.train(x_train,y_train,'LOO','c')
elapsed_time_train = time.time() - t
print(str(elm))
y_train_predicted = elm.predict(x_train)
print("Training time: %f" % elapsed_time_train)
print('Training Accuracy: ',(1-elm.error(y_train,y_train_predicted)))
y_test_predicted = elm.predict(x_test)
print('Test Accuracy: ',(1-elm.error(y_test,y_test_predicted)))
print(elm.confusion(y_test,y_test_predicted))
y_test_sk = y_test.argmax(1)
y_test_predicted_sk = y_test_predicted.argmax(1)
class_report = classification_report(y_test_sk, y_test_predicted_sk)
np.set_printoptions(precision=2)
print("Classification report\n: ", class_report)

# LOO ELM L1 Regularized
print('OP-ELM')
elm = hpelm.ELM(4,3, classification ="c")
elm.add_neurons(100,'sigm')
t = time.time()
elm.train(x_train,y_train,'LOO', 'OP','c')
elapsed_time_train = time.time() - t
print(str(elm))
y_train_predicted = elm.predict(x_train)
print("Training time: %f" % elapsed_time_train)
print('Training Accuracy: ',(1-elm.error(y_train,y_train_predicted)))
y_test_predicted = elm.predict(x_test)
print('Test Accuracy: ',(1-elm.error(y_test,y_test_predicted)))
print(elm.confusion(y_test,y_test_predicted))
y_test_sk = y_test.argmax(1)
y_test_predicted_sk = y_test_predicted.argmax(1)
class_report = classification_report(y_test_sk, y_test_predicted_sk)
np.set_printoptions(precision=2)
print("Classification report\n: ", class_report)

# CV ELM 5 fold
print('CV-ELM, 3 fold')
elm = hpelm.ELM(4,3, classification ="c")
elm.add_neurons(100,'sigm')
t = time.time()
e = elm.train(x_train,y_train,'CV','OP','c', k=5)
print('Error CV: ',e)
elapsed_time_train = time.time() - t
print(str(elm))
y_train_predicted = elm.predict(x_train)
print("Training time: %f" % elapsed_time_train)
print('Training Accuracy: ',(1-elm.error(y_train,y_train_predicted)))
y_test_predicted = elm.predict(x_test)
print('Test Accuracy: ',(1-elm.error(y_test,y_test_predicted)))
print(elm.confusion(y_test,y_test_predicted))
y_test_sk = y_test.argmax(1)
y_test_predicted_sk = y_test_predicted.argmax(1)
class_report = classification_report(y_test_sk, y_test_predicted_sk)
np.set_printoptions(precision=2)
print("Classification report\n: ", class_report)



# Validated ELM
print('Validate ELM')
# split data into train+validation set and test set
x_trainval, x_test, y_trainval, y_test = train_test_split(
    iris.data, iris.target, random_state=rnd_seed)
# split train+validation set into training and validation sets
x_train, x_valid, y_train, y_valid = train_test_split(
    x_trainval, y_trainval, random_state=1)
print("Size of training set: {}   size of validation set: {}   size of test set:"
      " {}\n".format(x_train.shape[0], x_valid.shape[0], x_test.shape[0]))

y_train = y_train.reshape(-1,1)
y_test= y_test.reshape(-1,1)
y_valid= y_valid.reshape(-1,1)
y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)
y_valid = onehot_encoder.fit_transform(y_valid)

elm = hpelm.ELM(4,3, classification ="c")
elm.add_neurons(100,'sigm')
t = time.time()
elm.train(x_train,y_train,'V','OP','c', Xv=x_valid, Tv=y_valid)
elapsed_time_train = time.time() - t
print(str(elm))
y_train_predicted = elm.predict(x_train)
print("Training time: %f" % elapsed_time_train)
print('Training Accuracy: ',(1-elm.error(y_train,y_train_predicted)))
y_test_predicted = elm.predict(x_test)
print('Test Accuracy: ',(1-elm.error(y_test,y_test_predicted)))
print(elm.confusion(y_test,y_test_predicted))
y_test_sk = y_test.argmax(1)
y_test_predicted_sk = y_test_predicted.argmax(1)
class_report = classification_report(y_test_sk, y_test_predicted_sk)
np.set_printoptions(precision=2)
print("Classification report\n: ", class_report)
