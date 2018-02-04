from loadiris import x_test, y_test, x_train, y_train

import time
import hpelm
import numpy as np

#metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# Reshape and apply OneHotEncoder to compute the 10class classifier
y_train = y_train.reshape(-1,1)
y_test= y_test.reshape(-1,1)
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False)
y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.fit_transform(y_test)

np.random.seed(10);

print('ELM\n')
model = hpelm.ELM(4,3, classification ="c")
model.add_neurons(500, 'sigm',  W = np.random.randn(4, 500))
model.add_neurons(3, 'lin')
print(str(model))
t = time.time()
model.train(x_train,y_train, 'c')
elapsed_time_train = time.time() - t
y_train_predicted = model.predict(x_train)

print("Training time: %f" % elapsed_time_train)
print('Training Error: ',model.error(y_train,y_train_predicted))
t = time.time()
y_test_predicted = model.predict(x_test)
elapsed_time_test = time.time() - t
print("Testing time: %f" % elapsed_time_test)
print('Test Error: ',model.error(y_test,y_test_predicted))

#print(model.confusion(y_test,y_test_predicted))

y_test_sk = y_test.argmax(1)
y_test_predicted_sk = y_test_predicted.argmax(1)
acc_score = accuracy_score(y_test_sk, y_test_predicted_sk)
cnf_matrix = confusion_matrix(y_test_sk, y_test_predicted_sk)
class_report = classification_report(y_test_sk, y_test_predicted_sk)
np.set_printoptions(precision=2)
print("Accuracy:\n", acc_score)
print("Confusion Matrix:\n", cnf_matrix)
print("Classification report\n: ", class_report)

print('ELM LOO')
t = time.time()

model.train(x_train,y_train, 'LOO' , 'c' )
elapsed_time_train = time.time() - t
print(str(model))
y_train_predicted = model.predict(x_train)

print("Training time: %f" % elapsed_time_train)
print('Training Error: ',model.error(y_train, y_train_predicted))
y_test_predicted = model.predict(x_test)
t=time.time()
elapsed_time_test = time.time() - t
print("Testing time: %f" % elapsed_time_test)
print('Test Error: ', model.error(y_test,y_test_predicted))

y_test_sk = y_test.argmax(1)
y_test_predicted_sk = y_test_predicted.argmax(1)
acc_score = accuracy_score(y_test_sk, y_test_predicted_sk)
cnf_matrix = confusion_matrix(y_test_sk, y_test_predicted_sk)
class_report = classification_report(y_test_sk, y_test_predicted_sk)
np.set_printoptions(precision=2)
print("Accuracy:\n", acc_score)
print("Confusion Matrix:\n", cnf_matrix)
print("Classification report\n: ", class_report)

print('ELM OP')
t = time.time()

model.train(x_train,y_train, 'OP' , 'c' )
elapsed_time_train = time.time() - t
print(str(model))
y_train_predicted = model.predict(x_train)

print("Training time: %f" % elapsed_time_train)
print('Training Error: ',model.error(y_train, y_train_predicted))
y_test_predicted = model.predict(x_test)
t=time.time()
elapsed_time_test = time.time() - t
print("Testing time: %f" % elapsed_time_test)
print('Test Error: ', model.error(y_test,y_test_predicted))

y_test_sk = y_test.argmax(1)
y_test_predicted_sk = y_test_predicted.argmax(1)
acc_score = accuracy_score(y_test_sk, y_test_predicted_sk)
cnf_matrix = confusion_matrix(y_test_sk, y_test_predicted_sk)
class_report = classification_report(y_test_sk, y_test_predicted_sk)
np.set_printoptions(precision=2)
print("Accuracy:\n", acc_score)
print("Confusion Matrix:\n", cnf_matrix)
print("Classification report\n: ", class_report)

