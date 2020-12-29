#import standard data sets 
from sklearn import datasets
#import the Logistic regression model 
from sklearn.linear_model import LogisticRegression
#split data set into a train and test set 
from sklearn.model_selection import train_test_split
#importing modules to measure classification performance 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix,accuracy_score
df=datasets.load_digits()

#df =datasets.load_wine() #load ’wine’ data set from standard data sets 
print(df)
#convert the DataFrame to a NumPy array
