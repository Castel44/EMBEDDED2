from sklearn import datasets
import matplotlib as plt
import numpy as np

iris = datasets.load_iris()

iris.keys()
iris.DESCR

iris["data"][:, 3]  # petal width, last attribute of dataset

y = (iris["target"] == 2).astype(np.int)

import pandas as pd

iris.data = pd.DataFrame(iris.data)

iris.data.describe()

# need to scale the dataset but first lets create a test set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Scaling data (mean 0, variance 1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
