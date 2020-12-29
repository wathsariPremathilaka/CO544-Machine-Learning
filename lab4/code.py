from sklearn import datasets
from sklearn.metrics import confusion_matrix,accuracy_score
import pandas as pd 
import numpy as np

from sklearn.datasets import load_wine

wine =datasets.load_wine() 
#print(wine)
wine_data=wine.data
#print(wine_data.shape)
#(178, 13) there are 178 samples with 13 features.

wine_labels=wine.target
#print(wine_labels.shape)
#(178,)

labels = np.reshape(wine_labels,(178,1))#reshaping the wine_labels to concatenate it with the wine_data
final_wine_data = np.concatenate([wine_data,labels],axis=1)#concatenate the data and labels along the second axis
#print(final_wine_data.shape)
#(178, 14)<-final shape of the array 

wine_dataset = pd.DataFrame(final_wine_data)#create the DataFrame of the final data 

features = wine.feature_names
#print(features)
#['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
#Here,label field is missing.Hence,manually add it to the features array 
features_labels = np.append(features,'label')
wine_dataset.columns = features_labels#embed the column names to the wine_dataset dataframe.

wine_dataset['label'].replace(0, 'class_0',inplace=True)
wine_dataset['label'].replace(1, 'class_1',inplace=True)
wine_dataset['label'].replace(2, 'class_2',inplace=True)

'''
print(wine_dataset.head())
   alcohol  malic_acid   ash  alcalinity_of_ash  magnesium  total_phenols  ...  proanthocyanins  color_intensity   hue  od280/od315_of_diluted_wines  proline    label
0    14.23        1.71  2.43               15.6      127.0           2.80  ...             2.29             5.64  1.04                          3.92   1065.0  class_0
1    13.20        1.78  2.14               11.2      100.0           2.65  ...             1.28             4.38  1.05                          3.40   1050.0  class_0
2    13.16        2.36  2.67               18.6      101.0           2.80  ...             2.81             5.68  1.03                          3.17   1185.0  class_0
3    14.37        1.95  2.50               16.8      113.0           3.85  ...             2.18             7.80  0.86                          3.45   1480.0  class_0
4    13.24        2.59  2.87               21.0      118.0           2.80  ...             1.82             4.32  1.04                          2.93    735.0  class_0
'''
from sklearn.preprocessing import StandardScaler
x = wine_dataset.loc[:, features].values
x = StandardScaler().fit_transform(x) # normalizing the features
#print(x.shape)
#(178, 13)

'''check whether the normalized data has a mean of zero and a standard deviation of one
print(np.mean(x)) -> 4.66735072755122e-16
print(np.std(x)) -> 1.0'''

feat_cols = ['feature'+str(i) for i in range(x.shape[1])] #convert the normalized features into a tabular format with the help of DataFrame.
normalised_wine = pd.DataFrame(x,columns=feat_cols)
'''
print(normalised_wine.tail())
 feature0  feature1  feature2  feature3  feature4  feature5  feature6  feature7  feature8  feature9  feature10  feature11  feature12
173  0.876275  2.974543  0.305159  0.301803 -0.332922 -0.985614 -1.424900  1.274310 -0.930179  1.142811  -1.392758  -1.231206  -0.021952
174  0.493343  1.412609  0.414820  1.052516  0.158572 -0.793334 -1.284344  0.549108 -0.316950  0.969783  -1.129518  -1.485445   0.009893
175  0.332758  1.744744 -0.389355  0.151661  1.422412 -1.129824 -1.344582  0.549108 -0.422075  2.224236  -1.612125  -1.485445   0.280575
176  0.209232  0.227694  0.012732  0.151661  1.422412 -1.033684 -1.354622  1.354888 -0.229346  1.834923  -1.568252  -1.400699   0.296498
177  1.395086  1.583165  1.365208  1.502943 -0.262708 -0.392751 -1.274305  1.596623 -0.422075  1.791666  -1.524378  -1.428948  -0.595160
'''
from sklearn.decomposition import PCA
pca_wine = PCA(n_components=3)
principalComponents_wine = pca_wine.fit_transform(x)

#create a DataFrame that will have the principal component values for all 178 samples.
principal_wine_Df = pd.DataFrame(data = principalComponents_wine, columns = ['principal component 1', 'principal component 2','principal component 3'])
'''
print(principal_wine_Df.tail())
  principal component 1  principal component 2  principal component 3
173              -3.370524              -2.216289              -0.342570
174              -2.601956              -1.757229               0.207581
175              -2.677839              -2.760899              -0.940942
176              -2.387017              -2.297347              -0.550696
177              -3.208758              -2.768920               1.013914
'''

import matplotlib.pyplot as plt 

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(211, projection='3d',title="Principal Component Analysis of Wine Dataset")

ax.set_xlabel('Principal Component 1',fontsize=10, labelpad=30)
ax.set_ylabel('Principal Component 2',fontsize=10,labelpad=30)
ax.set_zlabel('Principal Component 3',fontsize=10,labelpad=20)


targets = ['class_0', 'class_1', 'class_2']
colors = ['r', 'g', 'b']

for target, color in zip(targets,colors):
    indicesToKeep = wine_dataset['label'] == target
    ax.scatter(principal_wine_Df.loc[indicesToKeep, 'principal component 1'],principal_wine_Df.loc[indicesToKeep, 'principal component 2']
               , principal_wine_Df.loc[indicesToKeep, 'principal component 3'], c = color, s = 50)

ax.legend(targets,prop={'size': 10},loc='upper left')
plt.show()