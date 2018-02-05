import time
import numpy as np
from loadMNIST import x_test,y_test,x_train,y_train


from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

print("SVM classifier")

svc_clf = SVC(C=10.0, kernel="rbf", degree=4, gamma="auto", coef0=0.0 )
t = time.time()
svc_clf.fit(x_train, y_train)
elapsed_time_train = time.time() - t

# Predict value
t = time.time();
Y_prediction = svc_clf.predict(x_test)
elapsed_time_test = time.time() - t
print("Training time: %f" % elapsed_time_train)
print("Testing time: %f" % elapsed_time_test)
acc_score = accuracy_score(y_test, Y_prediction)
cnf_matrix = confusion_matrix(y_test, Y_prediction)
class_report = classification_report(y_test, Y_prediction)
np.set_printoptions(precision=2)
print("Accuracy:\n", acc_score)
print("Confusion Matrix:\n", cnf_matrix)
print("Classification report\n: ", class_report)
