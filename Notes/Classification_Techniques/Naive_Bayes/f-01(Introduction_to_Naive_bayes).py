from sklearn import datasets
from sklearn import model_selection
# Importing the multinomial and Gaussian Naive Bayes..........
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix


# Loading_Iris_Dataset......................
iris = datasets.load_iris()
X = iris.data
Y = iris.target

#Splitting Dataset .............
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.25, random_state=0)


# Using the Gaussian Kernel for Naive Bayes.................
clf = GaussianNB()
# Attributes of gaussian object:
# class_prior_: Probability of each class
# class_count_: number of training samples per class
# theta_: mean of each feature per class
# sigma_: variance of each feature per class
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
print(classification_report(Y_test,Y_pred))
print(confusion_matrix(Y_test,Y_pred))

# Using Multinomial Naive Bayes............................
clf = MultinomialNB()
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
print(classification_report(Y_test,Y_pred))
print(confusion_matrix(Y_test,Y_pred))