import numpy as np
import matplotlib.pyplot as plt
X=np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11]])
plt.scatter(X[:,0], X[:,1])
plt.show()
from sklearn.cluster import KMeans
# Some parameters of Kmeans:
# 1. n_cluster- Number of clusters to find; by default n_clusters=8
# 2. precompute_distances='auto', here to save time it uses differneces of Mean and data point in next iteration
# 3. init= how to select first k random points, by default it uses kmeans++(algorithm tries to select far away points)
# 4. n_init= How many times we run the Kmeans algorithm, with different initlization; The function with minimal loss value is selected
# 5. max_iter- No of iterations to run in one run
# 6. 
kmeans=KMeans(n_clusters=2)
kmeans.fit(X)
#  Which point got pushed to which cluster......
kmeans.labels_
# Mean values of clusters................
kmeans.cluster_centers_
#  Plotting Data Points and Clusters and Means 
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1])
plt.show()
# Trying Same data with different K (number of clusters)...................
kmeans=KMeans(n_clusters=3)
kmeans.fit(X)
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1])
plt.show()


