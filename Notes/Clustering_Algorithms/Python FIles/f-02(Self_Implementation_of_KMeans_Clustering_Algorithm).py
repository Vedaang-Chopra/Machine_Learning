import numpy as np
def fit (data, k = 2, max_iter = 100):
        means = []
        # randomly initialize the means
        for i in range(k):
            means.append(data[i])
        for i in range(max_iter):
            # assign the data points to the cluster that they belong to
            # create empty clusters
            clusters = []
            for j in range(k):
                clusters.append([])
            for point in data:
                # find distance to all the mean values
                distances = [((point - m)**2).sum() for m in means]
                # find the min distance
                minDistance = min(distances)
                # find the mean for which we got the minimum distance --> l
                l = distances.index(minDistance)
                # add this point to cluster l
                clusters[l].append(point)

            # calculate the new mean values
            change = False
            for j in range(k):
                new_mean = np.average(clusters[j], axis=0)
                if not np.array_equal(means[j], new_mean):
                    change = True
                means[j] = new_mean
            if not change:
                break
        return means

def predict(test_data, means): 
        predictions = []
        for point in test_data:
           # find distance to all the mean values
            distances = [((point - m)**2).sum() for m in means]
            # find the min distance
            minDistance = min(distances)
            # find the mean for which we got the minimum distance --> l
            l = distances.index(minDistance)
            # add this point to cluster l
            predictions.append(l)
        return predictions
from sklearn.datasets import load_iris
iris=load_iris()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(iris.data,iris.target)
means=fit(x_train)
predict(x_test, means)
