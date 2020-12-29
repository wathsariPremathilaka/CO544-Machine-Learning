import  numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

df = pd.read_csv('Boston_Housing.csv')#import the data set

train,test= train_test_split(df, test_size= 0.20, random_state=1)#Split 80% of data as the training set and rest as the test set.
print(len(test))

train1=pd.DataFrame(train)#set as a dataframe

def create_samples():#create a function to Create 50 samples (number of instances, n=100) by sampling with replacement
	sample=train1.sample(n=100,replace=True)
	return sample


test=test.values#convert test set into numpy array
x_test,y_test = test[:,0:3] ,test[:,3]#Divide test set into feature matrix and response matrix
x_test= np.c_[np.ones(98), x_test]#arrange x by settiing x[0]=1 as given x array



response_variable_values=[] #define a array to store response variable values for test data

i=0
while i<50:#loop for create 50 samples and find the corresponding predictions for test set
  
  sample=create_samples()
  sample=sample.values

  x,y = sample[:,0:3] , sample[:,3]
  x= np.c_[np.ones(100),x]
  x_transpose=np.transpose(x) 
  xt_x=x_transpose.dot(x)
  xt_xInverse=np.linalg.inv(xt_x)
  xt_y=x_transpose.dot(y)
  beta_hat=xt_xInverse.dot(xt_y)#calculate beta hat
  response_variable_values.insert(i,(x_test.dot(beta_hat)))#calculate response value and insert tem into the above defined array
  i=i+1

average_predictions=sum(response_variable_values)/len(response_variable_values)# Make the predictions for test data, by taking the average 


plt.scatter(average_predictions, y_test,color = "b", marker = ".", s = 60) #Visualize the residual error using a scatter plot.
plt.title('Residual errors') 
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')
plt.show()
  