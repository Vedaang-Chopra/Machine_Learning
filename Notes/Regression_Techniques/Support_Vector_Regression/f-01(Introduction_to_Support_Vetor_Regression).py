# Using Support Vector Machine for Regression Problems.....................
from sklearn import model_selection
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import GridSearchCV


# Loading The Boston Dataset.........................
boston=datasets.load_boston()                       # Loading the Boston House Pricing Data set.

# Splitting the Data into training and testing data.........................
x_train,x_test,y_train,y_test=model_selection.train_test_split(boston.data,boston.target)


# Using Support Vector Regressor with Gaussian Kernel...................................
algo=svm.SVR(kernel='rbf')
grid = {'C' : [1e2, 1e3, 5e3, 1e4, 5e4, 1e5],
       'gamma' : [1e-3, 5e-4, 1e-4, 5e-3]}

abc = GridSearchCV(algo, grid)
abc.fit(x_train, y_train)
# print(type(abc.best_estimator_))
print('Score for SVR with gaussian kernel:',abc.best_estimator_.score(x_test,y_test))


# Using Support Vector Regressor with Linear Kernel.................
algo_1=svm.SVR(kernel='linear')
grid = {'C' : [1e2, 1e3, 5e3, 1e4, 5e4, 1e5],
       'gamma' : [1e-3, 5e-4, 1e-4, 5e-3]}
abc_1 = GridSearchCV(algo_1, grid)
abc_1.fit(x_train, y_train)
print('Score for SVR with linear kernel:',abc_1.best_estimator_.score(x_test,y_test))
