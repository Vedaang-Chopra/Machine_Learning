# Applying PCA on two dimensional dummy data .............................

# PCA focuses only the data and not the output labels.
# Note: Reduction is always dependent on data. Which features to reduce, and how much information will be lost is dependent on the dataset.
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


# Using PCA on 2-Dimensional Data.....................
# Plotting random/dummy 2-D data..........
x1 = np.array([1,2,3,4,5,6])
x2 = np.array([7.5, 11, 16, 18, 20, 26])
plt.scatter(x1, x2)
plt.show()

# Concatenating the two single feature arrays for 2-D data. It is done to create a two dimensional dataset(x1, x2 as features).
X = np.c_[x1, x2]           # Concatenate function,combines two lists.

# Creating the PCA object.....
# n_components: This parameter specifies how many features we need finally. If n_components is not passed then the no. of new dimensions is same as no of original features.
# Here we can try any number of feature values less than equal to the total number of features we originally had.
pca = PCA(n_components = 1)

# Here when we fit we try to find a line that finds a best line for the data.
# With transform function we transform the data according to x1' and x2' features. These new values are according to new directions.
X_reduced = pca.fit_transform(X)
print(X_reduced)
# The compnents_ parameter tells us about the unit vector in the new direction(basically i-cap and j-cap values (because originally its two dimensional data) for the x1' and x2' directions.)
# These are unit vectors along the new features which hold maximum variance. This is sorted and the first direction holds the information with maximum variance.
print(pca.components_)

# To bring back the data to original dimensions/directions we can use the inverse_transform function.
# If we reduce the feature values we will not be able to get back our data as some information is lost. The values will differ from original values.
X_approx = pca.inverse_transform(X_reduced)

#  Plotting the new x values after reducing in the original direction. After reducing components the point lie on a line as one direction is reduced.
plt.plot(X_approx[:, 0], X_approx[:, 1])
plt.scatter(x1, x2)
plt.show()

