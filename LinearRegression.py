import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

df= pd.read_csv("insurance.csv")
print(df.head())
# print(df.info())
print(df.isnull().sum())

# label encoder
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])       # male/female -> 0/1
df['smoker'] = le.fit_transform(df['smoker']) # yes/no -> 1/0


target='charges'
df['smoker_bmi'] = df['smoker'] * df['bmi']
df['age_smoker_bmi'] = df['age'] * df['smoker'] * df['bmi']
features=['age', 'sex', 'bmi', 'children', 'smoker', 'region','smoker_bmi','age_smoker_bmi']
x= df[features].values
y= df[target]
#preprocessing 

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [5])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

#Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#Feature Scaling
scaler= StandardScaler()
x_train= scaler.fit_transform(x_train)
x_test= scaler.transform(x_test)
y_train=scaler.fit_transform(y_train.values.reshape(-1,1)).flatten()
y_test=scaler.transform(y_test.values.reshape(-1,1)).flatten()

#Training model
model= LinearRegression()
model.fit(x_train, y_train)

#Predicting test set results
y_pred= model.predict(x_test)
mse= mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

r2= r2_score(y_test,y_pred)
print("R2 Score:",r2)

#Visualizing Actual vs Predicted
min_val= min(min(y_test), min(y_pred))
max_val= max(max(y_test), max(y_pred))
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Medical Charges")
plt.plot([min_val, max_val], [min_val, max_val])
plt.show()

