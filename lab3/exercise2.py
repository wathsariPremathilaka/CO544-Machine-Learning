#import standard data sets 
from sklearn import datasets
#import the Logistic regression model 
from sklearn.linear_model import LogisticRegression
#split data set into a train and test set 
from sklearn.model_selection import train_test_split
#importing modules to measure classification performance 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix,accuracy_score

df =datasets.load_digits() #load digits data set from standard data sets 

x=df["data"] #defining features values 
y =df["target"] #defining target variable values


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)#splitting data set into a train and test set with 80% and 20% 

log_reg = LogisticRegression() #creating an instance of the model 
log_reg.fit(x_train,y_train) #fitting the relationship between data
predictions = log_reg.predict(x_test) #predict labels for test data

print(confusion_matrix(y_test, predictions)) #find confusion matrix
print(accuracy_score(y_test, predictions)) #find accuracy_score