import numpy as np
import pandas as pd 

#importing dataset
dataset_train=pd.read_csv("Train.csv")
dataset_test=pd.read_csv("Test.csv")

x_train=dataset_train.iloc[:,:-1].values
y_train=dataset_train.iloc[:,5].values
y_train = np.reshape(y_train, (-1,1))
x_test=dataset_test.iloc[:,:].values

#Fitting Multiple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#predicting the test set results using linear regression model we created above
y_pred=regressor.predict(x_test)

#building the optimal model using backward elimination
import statsmodels.api as sm

x_opt= x_train[:,[0,1,2,3,4]]
regressor_ols=sm.OLS(endog=y_train ,exog=x_opt).fit()
regressor_ols.summary()

y_pred=regressor_ols.predict(x_test)
