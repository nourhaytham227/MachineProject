import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

df= pd.read_csv("insurance.csv")
print(df.head())
print(df.info())

target='charges'
features=['age', 'sex', 'bmi', 'children', 'smoker', 'region']
x= df[features].values
y= df[target]
print(x)
print(y)