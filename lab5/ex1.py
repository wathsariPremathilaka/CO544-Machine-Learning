import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from mpl_toolkits import mplot3d
from sklearn.cluster import KMeans

#(a)import iris dataset from sklearn library and make the dataset into an unlabeled dataset
iris = datasets.load_iris()
X = iris.data 
y = iris.target

#(b)Find the optimal k
wcss = [] #within cluster sum of squares 
for i in range(1, 10): 
	kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0) 
	kmeans.fit(X) 
	wcss.append(kmeans.inertia_) 
plt.plot(range(1, 10), wcss) 
plt.title('Elbow Method') 
plt.xlabel('Number of clusters') 
plt.ylabel('WCSS') 

#From the plot,We can observe that the “elbow” is the number 3 which is optimal for this case. 
#Therfore, the optimal k is 3.

#(c) Fit the k-Means algorithm with the k value identiﬁed from part(b). 
k_means = KMeans(n_clusters=3)
k_means.fit(X)
k_means_predicted = k_means.predict(X)

#(d) Explain the results of following code snippet.
'''
kmeans.cluster_centers_
This code snippet use for finding the center of the clusters


print(k_means.cluster_centers_)

This is the result of the value of the centroids
[[5.9016129  2.7483871  4.39354839 1.43387097]
 [5.006      3.428      1.462      0.246     ]
 [6.85       3.07368421 5.74210526 2.07105263]]
'''

#(e) Visualize the data points and cluster centers in a 3D plot 
centroids = k_means.cluster_centers_
target_names = iris.target_names
colors = ['navy', 'turquoise', 'darkorange']
plt.figure('K-Means and centroids on Iris Dataset', figsize=(7,7))
ax = plt.axes(projection = '3d',title='K-Means and centroids on Iris Dataset')
ax.scatter(X[:,0],X[:,1],X[:,2], c=y , cmap='Set2', s=50)

ax.scatter(centroids[0,0],centroids[0,1],centroids[0,2] ,c='r', s=200, label='centroid',marker='*')
ax.scatter(centroids[1,0],centroids[1,1],centroids[1,2] ,c='r', s=200,marker='*')
ax.scatter(centroids[2,0],centroids[2,1],centroids[2,2] ,c='r', s=200,marker='*')
ax.legend()
ax.set_xlabel('0th variable')
ax.set_ylabel('1st variable')
ax.set_zlabel('2nd variable')

plt.show()
