import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('./Machine Learning A-Z/Part 1 - Data Preprocessing/Data.csv')
print(dataset)

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:, 3].values