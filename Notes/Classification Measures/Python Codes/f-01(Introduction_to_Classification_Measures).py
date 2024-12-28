from sklearn import datasets
from sklearn import model_selection
# Importing the Confusion Matrix and Classification Report.........................
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm


# Loading_Iris_Dataset......................
iris = datasets.load_iris()
X = iris.data
Y = iris.target

# Splitting the Dataset .............
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.25, random_state=0)
clf=svm.SVC(kernel='linear')
clf.fit(X_train,Y_train)
y_pred=clf.predict(X_test)
y_pred_train=clf.predict((X_train))

# Confusion Matrix is used to check how confused our model is. It finds the correct and incorrect predicted values for each class.
print('Confusion Matrix For Testing Data:')
print(confusion_matrix(Y_test,y_pred))
print('Confusion Matrix For Training Data:')
print(confusion_matrix(Y_train,y_pred_train))


# Classification report is the Collection of precision, recall, f1-score and support metrics for each class.
# Used to analyse the predictions and confusion matrix.

print('Classification Report of Testing Data:')
print(classification_report(Y_test,y_pred))
print('Classification Report of Training Data:')
print(classification_report(Y_train,y_pred_train))
