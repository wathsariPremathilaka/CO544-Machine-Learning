


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from numpy.linalg import inv

# Import the Boston Housing.csv ﬁle
df = pd.read_csv('Boston_Housing.csv')

# Split 80% of data as the training set and rest as the test set
train, test = train_test_split(df, test_size=0.2, random_state=1)

# set dataset as a dataframe
train_new= pd.DataFrame(train)

# create 50 samples (number of instances n =100) by sampling with replacement(boostrap)
def create_samples():
    sample = train_new.sample(n=100, replace=True)
    return sample


# convert test set into numpy array
test = test.values

# divide test set into feature matrix and response matrix
x_test, y_test = test[:, 0:3], test[:, 3]

# adding a column as first column with all elements
x_test = np.c_[np.ones(98), x_test]

# define an array to store responses values for particular test data
response_values=[]

#  create a loop for 50 samples and find the particular predictions
i=0
while i < 50:
    sample = create_samples()
    sample = sample.values
    
    x, y = sample[:, 0:3], sample[:, 3]
    x = np.c_[np.ones(100), x]

    x_T = np.transpose(x) # get the transpose matrix of x
    xTx = np.dot(x_T, x)  # get the multiplication of x transpose and x
    a1 = np.array(xTx) # get the inverse of xTx
    xTx_inv = inv(a1)
    x_Ty = np.dot(x_T, y) # get the multiplication of x_T and y
    b = np.dot(xTx_inv,x_Ty) # calculate the parameter value
    response_values.insert(i, (np.dot(x_test,b)))
    i = i+1

# make the predictions for the test data  by taking the average
avg_pred = sum(response_values)/len(response_values)
print(len(avg_pred))
plt.scatter(avg_pred, y_test,color = "b", marker = ".", s = 60)#residual errors plotting

# visualize the residual errors for both train and test data sets
plt.title('Residual errors')
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')
plt.show()





