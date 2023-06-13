from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

TRAIN_PATH = '/Users/sribhav/PycharmProjects/playground-series-s3e16/train.csv'
dataset = pd.read_csv(TRAIN_PATH)
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values
x1 = X[:, 0]
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer

onehot = OneHotEncoder()
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False), [0])], remainder='passthrough')
X = ct.fit_transform(X)

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, Y)

TEST_PATH = '/Users/sribhav/PycharmProjects/playground-series-s3e16/test.csv'


test_dataset = pd.read_csv(TEST_PATH)
y_test = test_dataset.iloc[1:].values
y_test = ct.fit_transform(y_test)

lin_reg2.predict(poly_reg.fit_transform(y_test))
