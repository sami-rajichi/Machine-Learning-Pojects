import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data"
data = pd.read_csv(url, names=['vendor name', 'Model Name', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP'])
#data = data[['MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP']]
cols = ['vendor name', 'Model Name']
data = data.drop(cols, 1)

print(data.head())

x = data.iloc[:, :-1].values
y = data.iloc[:, 7].values

from sklearn.model_selection import train_test_split
x_test, x_train, y_test, y_train = train_test_split(x, y, test_size=0.2)

from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(x_train, y_train)

predictions = linear.predict(x_test)
for i in range(len(predictions)):
  print(predictions[i], x_test[i], y_test[i])