import  numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

df = pd.read_csv('Boston_Housing.csv')#import the data set

#convert the DataFrame to a NumPy array
df = df.values

def create_samples():
	sample=train.sample(n=5,replace=True)
	return sample

# Seperate features and response into separate variables
#x,y = df[:,0:3] , df[:,3]


# Split into train and test sets
from sklearn.model_selection import train_test_split
train,test= train_test_split(df, test_size= 0.20, random_state=15)

x_train,y_train = train[:,0:3] , train[:,3]
x_test,y_test = test[:,0:3] ,test[:,3]



b=[]
i=0
while i<50:
	sample=create_samples()
	#print(sample)
	sample=sample.values
	#print(sample)
	x,y = sample[:,0:3] , sample[:,3]
	#print(x)
	x_transpose=np.transpose(x) 
	xt_x=x_transpose.dot(x)
	xt_xInverse=np.linalg.inv(xt_x)
	xt_y=x_transpose.dot(y)
	beta_hat=xt_xInverse.dot(xt_y)
	#print(beta_hat)
	#b[1]=
	b.insert(i,(x.dot(beta_hat)))
	i=i+1
print(b[0])