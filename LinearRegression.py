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
#print(df.head())
#print(df.info())

target='charges'
features=['age', 'sex', 'bmi', 'children', 'smoker', 'region']
x= df[features].values
y= df[target]

#preprocessing 
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1, 4, 5])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

#Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#Feature Scaling
scaler= StandardScaler()
x_train= scaler.fit_transform(x_train)
x_test= scaler.transform(x_test)

#Training model
model= LinearRegression()
model.fit(x_train, y_train)

#Predicting test set results
y_pred= model.predict(x_test)
mse= mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

#Visualizing Actual vs Predicted
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Medical Charges")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.show()

