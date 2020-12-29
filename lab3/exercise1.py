import  numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split


df = pd.read_csv('Boston_Housing.csv')#import the data set

train,test= train_test_split(df, test_size= 0.20, random_state=1)#Split 80% of data as the training set and rest as the test set


train=train.values#convert test and train set into numpy array
test=test.values

x_train,y_train=train[:,0:3] ,train[:,3]#Divide test set into feature matrix and response matrix
x_test,y_test = test[:,0:3] ,test[:,3]

x_train= np.c_[np.ones(391),x_train]#arrange x by settiing x[0]=1 as given x array
x_test= np.c_[np.ones(98), x_test]


#find transpose
xtrain_transpose=np.transpose(x_train) 

#find matrix multiplication of x transpose and x 
xt_x=xtrain_transpose.dot(x_train)

#find inverse of the above multiplication
xt_xInverse=np.linalg.inv(xt_x)

#find multification of the x transpose and y
xt_y=xtrain_transpose.dot(y_train)


#Find the parameter value(beta_hat)
beta_hat=xt_xInverse.dot(xt_y)


#Predict the response values for both test and train set
y_hat=x_train.dot(beta_hat)
y_hat_test=x_test.dot(beta_hat)


residual_error_train=y_train-y_hat
residual_error_test=y_test-y_hat_test
#print(residual_error)

plt.title('Residual errors') 
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')

plt.scatter(y_hat, y_train,color = "b", marker = ".", s = 60) #plotting residual errors for training set
plt.scatter(y_hat_test, y_test,color = "r", marker = "*", s = 60) #plotting residual errors for test set
plt.show() #displaying the plot