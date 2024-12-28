# Applying PCA on three dimensional dummy data .............................

from mpl_toolkits.mplot3d import Axes3D, proj3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generating Random 3-Dimensional Data...........................
# The random seed ensures that the random is same every time we run the code


np.random.seed(2343243)

# We are using a distribution function by providing the mean and covariance matrix. These are just random values that we have selected.
mean_vec1 = np.array([0,0,0])
cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
class1 = np.random.multivariate_normal(mean_vec1, cov_mat1, 100)

mean_vec2 = np.array([1,1,1])
cov_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
class2 = np.random.multivariate_normal(mean_vec2, cov_mat2, 100)


# Plotting the 3-Dimensional Data...............
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111,projection='3d')
ax.plot(class1[:, 0], class1[:, 1], class1[:, 2], 'o')
ax.plot(class2[:, 0], class2[:, 1], class2[:, 2], '^')
plt.show()

# Concatinating the Three Fetaures to create a dataframe.....................
all_data = np.concatenate((class1, class2))

# Creating a PCA Object........
# Reducing 3-Dimensional data into 2-Dimensional data
pca = PCA(n_components = 2)
transformed_data = pca.fit_transform(all_data)

print(transformed_data)
print(pca.components_)

plt.plot(transformed_data[0:100,0],transformed_data[0:100,1],"o")
plt.plot(transformed_data[100:200,0],transformed_data[100:200,1],"^")
plt.show()


X_approx = pca.inverse_transform(transformed_data)

# Plotting data values after inversing the tranformed data to check the change in values.....................
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111,projection='3d')
ax.plot(X_approx[:, 0], X_approx[:, 1], X_approx[:, 2], '^')
plt.show()


# Checking whether the data lies in a plane or not.
a = -0.409689
b = 7.2827
c = - 7.1008
i = 10
a * X_approx[i][0] + b* X_approx[i][1] + c * X_approx[i][2]