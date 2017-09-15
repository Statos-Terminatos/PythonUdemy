import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('./Machine Learning A-Z/Part 1 - Data Preprocessing/Data.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 3].values

from sklearn.preprocessing import Imputer
# strategy correspond the method
imputer = Imputer(missing_values = "NaN", axis=0, strategy="mean")

# To fit the matrix
imputer = imputer.fit(X[:,1:3]) # the upper bond is not included, index starts at zero
X[:,1:3] = imputer.transform(X[:,1:3])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(y)