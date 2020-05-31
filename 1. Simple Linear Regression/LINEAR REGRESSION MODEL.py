import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

#importing dataset
dataset_x_train=pd.read_csv("Linear_X_Train.csv")
dataset_y_train=pd.read_csv("Linear_Y_Train.csv")
dataset_x_test=pd.read_csv("Linear_X_Test.csv")

x_train=dataset_x_train.iloc[:,:].values
y_train=dataset_y_train.iloc[:,:].values
x_test=dataset_x_test.iloc[:,:].values

#fitting simple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#predicting the test set results using linear regression model we created above
y_pred=regressor.predict(x_test)

#visualing training set result
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train),color='blue')
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#visualing test set result
plt.scatter(x_test, y_pred, color='red')
plt.plot(x_train, regressor.predict(x_train),color='blue')
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
