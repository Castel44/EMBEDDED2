import time
import numpy as np
from Misc.loadMNIST import x_test, y_test, x_train, y_train

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# simple RandomForestClassifier examples for MNIST dataset classification

# NO CROSS VALIDATION EXAMPLE

print('FOREST')

# standard values for classifier

forest_clf = RandomForestClassifier(random_state=42)
t = time.time()
forest_clf.fit(x_train, y_train)
elapsed_time_train = time.time() - t

# Predict value
t = time.time();
Y_prediction = forest_clf.predict(x_test)
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

# cross validation performance to check if the model has overfitted

from sklearn.model_selection import cross_val_score

accuracy_cv = cross_val_score(forest_clf, x_train, y_train, scoring="accuracy", cv=4, verbose=0)
forest_clf

# Display accuracy for cv on train set
print("Accuracy:", accuracy_cv)
print("Mean Accuracy:", accuracy_cv.mean())
print("Std dev Accuracy:", accuracy_cv.std())

# note : mean accuracy score is higher here than the accuracy that has been computed for the full training set
#        this is probably because the model is trained and evaluated on less examples. This actually tells the model is not overfitting
#        For example setting cv = 2 the training set is splitted into two sets and cross evaluation is performed. This yields a value
#        for the mean accuracy near the one obtained for the full dataset. Increasing cv makes initially the mean accuracy higher, then lower and ultimately
#        for very high values higher again than full dataset accuracy. For very high values cross validation score is in fact extremely biased
#        because the dataset has been splitted into training and test sets of extremely small size.
#        Still the two performance metrics should not be confused. The model used in fact is the same actually.

print("check if the model has overfit the data")
input("Press Enter to continue...")
# %% check if the model is overfitting

# fine-tuning model with gridsearch

from sklearn.model_selection import GridSearchCV

param_grid = [{'n_estimators': [10, 20, 50], 'max_features': [8, 16, 32]}
              # {'bootstrap': [False], 'n_estimators': [30, 100, 150], 'max_features' : [64]},
              ]

grid_search = GridSearchCV(forest_clf, param_grid, cv=10, scoring="accuracy", verbose=0)

t = time.time();
grid_search.fit(x_train, y_train)
elapsed_time_train = time.time() - t

# commented code to check best params and error obtained for each parameter

# grid_search.best_params_

# grid_search.best_estimator_

# cvres= grid_search.cv_results_

# for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#    print(acc_score, params)

print('\n\nFOREST- GRID SEARCH')

post_grid_clf = grid_search.best_estimator_

# post grid prediction

t = time.time();
Y_prediction = post_grid_clf.predict(x_test)
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
