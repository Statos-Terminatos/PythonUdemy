import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('./Machine Learning A-Z/Part 1 - Data Preprocessing/Data.csv')

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:, 3].values

from sklearn.preprocessing import Imputer
# strategy correspond the method
imputer = Imputer(missing_values = "NaN", axis=0, strategy="mean")

# To fit the matrix
imputer = imputer.fit(X[:,1:3]) # the upper bond is not included, index starts at zero
X[:,1:3] = imputer.transform(X[:,1:3])

print(X)