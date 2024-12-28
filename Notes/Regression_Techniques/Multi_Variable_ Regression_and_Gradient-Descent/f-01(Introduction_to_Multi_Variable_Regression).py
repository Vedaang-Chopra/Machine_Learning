# This code is an example of multi-variable expression. Here we will add features (having quadratic and cubic relation to the exsisting features)
# to the boston dataset and see the change in the score and if it increases we will add to the main dataset.

import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import datasets
from sklearn.linear_model import LinearRegression

# Loading the Boston Dataset........................

boston=datasets.load_boston()                       # Loading the Boston House Pricing Data set.
x=boston.data
y=boston.target

dfx=pd.DataFrame(x)
dfy=pd.DataFrame(y)
# print(boston.feature_names)
dfx.columns=boston.feature_names                    # Changing the data frame headers into feature names
# print(dfx.describe())

# Splitting the data into training and testing..........................
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,random_state=0)

# Applying Linear Regression on the Dataset..................................
algo=LinearRegression()
algo.fit(x_train, y_train)                                # Now we have our algorithm we train it using the fit function which takes x_train and y_train.
# print(algo.score(x_test,y_test))
score=algo.score(x_test,y_test)
print('Score with Linear Regression:',score)

algo_1 = LinearRegression()
for i in range(0,len(boston.feature_names)):
    # print(i)
    dfx[boston.feature_names[i]+'_'+boston.feature_names[i]]=dfx[boston.feature_names[i]]**2
    dfx[boston.feature_names[i] + '_' + boston.feature_names[i]+ '_' + boston.feature_names[i]] = dfx[boston.feature_names[i]] **3
    if np.isnan(dfx[boston.feature_names[i]+'_'+boston.feature_names[i]].values).any()==True or np.isinf(dfx[boston.feature_names[i]+'_'+boston.feature_names[i]].values).any()==True :
        dfx.drop(boston.feature_names[i] + '_' + boston.feature_names[i], axis=1, inplace=True)
    if np.isnan(dfx[boston.feature_names[i] + '_' + boston.feature_names[i]+ '_' + boston.feature_names[i]].values).any() == True or np.isinf(
            dfx[boston.feature_names[i] + '_' + boston.feature_names[i]+ '_' + boston.feature_names[i]].values).any() == True:
        dfx.drop(boston.feature_names[i] + '_' + boston.feature_names[i]+ '_' + boston.feature_names[i], axis=1, inplace=True)

    x1 = dfx.values
    x_train_1, x_test_1, y_train_1, y_test_1 = model_selection.train_test_split(x1, y, random_state=0)
    algo_1.fit(x_train_1,y_train_1)
    # print('Original_score:',score)
    # print('Score with new column:',algo_1.score(x_test_1,y_test_1))
    if score>algo_1.score(x_test_1, y_test_1):
        dfx.drop(boston.feature_names[i]+'_'+boston.feature_names[i],axis=1,inplace=True)
        dfx.drop(boston.feature_names[i] + '_' + boston.feature_names[i] + '_' + boston.feature_names[i], axis=1,
                 inplace=True)
    else:
        score=algo_1.score(x_test_1,y_test_1)
        continue
print('Final Score with Quadratic and Cubic Variables:',score)






