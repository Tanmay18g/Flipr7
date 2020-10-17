# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from sklearn.metrics import mean_squared_error
import numpy as np
# Create our own function to predict accuracy
def custom_accuracy(y_test,y_pred,thresold):
    right = 0

    l = len(y_pred)
    for i in range(0,l):
        if(abs(y_pred[i]-y_test[i]) <= thresold):
            right += 1
        else:
            print(y_pred[i],y_test[i])
    return ((right/l)*100)

#
def square(y_test,y_pred):
    ans=0
    for i in range (0,len(y_pred)):
        ans+=pow(abs(y_pred[i]-y_test[i]),2)
    return ans
def squaremean(y_test,y_pred):
    ans=0
    for i in range (0,len(y_pred)):
        ans+=pow(abs(y_pred[i]-y_test[i]),2)
    return ans

# def decisiontree():

import pandas as pd
# Importing the dataset
dataset = pd.read_excel('Data.xlsx', sheet_name='Train_Data')
# print(dataset)
# print(dataset.info())
# Typecast Hs to string
dataset['HS']=dataset['HS'].astype(str)
# create new column Batsman as a part of feature engineering
dataset['Batsman'] = 0
for i in range(0,100):
    p=dataset['HS'][i]
    if p[-1]=='*':
        if int(p[:-1])>30:
            dataset['Batsman'][i]=1
        dataset['HS'][i]=p[:-1]
# 3 values in Avg is null to check this and put values in it
for i in range (0,100):
    try:
        float(dataset['Avg'][i])
    except:
        # print("error")
        if (dataset['Inns'][i]-dataset['NO'][i]) == 0:
            dataset['Avg'][i] = dataset['2018_Runs'][i]/dataset['Inns'][i]
        else:
            dataset['Avg'][i]=(dataset['2018_Runs'][i]/(dataset['Inns'][i]-dataset['NO'][i]))
        # print(dataset['Avg'][i])

# print(dataset['Avg'])
# typecast Avg to float
dataset['Avg']=dataset['Avg'].astype(float)
# print(dataset['Batsman'])
dataset['HS']=dataset['HS'].astype(int)
print(dataset['HS'].dtype)
dataset['BON'] = 0
# create new variable BON
for i in range (0,100):
    if dataset['2018_Runs'][i] > 100:
        dataset['BON'][i] = 1

X = dataset.iloc[:,[2,4,6,7,10,11,14,15]].values
y = dataset.iloc[:, 13].values
# print(y)
# print(dataset.describe())
#
#
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=2)

# Feature Scaling
# Scale out input variables

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the dataset

# Testing the dataset on trained model
# y_pred = lin.predict(X_test)
# score = lin.score(X_test,y_test)*100
# print("R square value:" , score)
# print("linear Custom accuracy:" , custom_accuracy(y_test,y_pred,20))
# print("Linear square: ", square(y_test,y_pred))
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print("Linear RMSE: %f" % (rmse))
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=50,max_depth=4, max_leaf_nodes=25, random_state=10)
reg.fit(X_train,y_train)

# Testing the dataset on trained model
y_pred1 = reg.predict(X_test)
score = reg.score(X_test,y_test)*100
print("R square value:" , score)
print("Random Custom accuracy:" , custom_accuracy(y_test,y_pred1,20))
print("Random square: ", square(y_test,y_pred1))
rmse = np.sqrt(mean_squared_error(y_test, y_pred1))
print("Random RMSE: %f" % (rmse))
# from sklearn.svm import SVR
# svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
# y_rbf_pred2 = svr_rbf.fit(X_train, y_train).predict(X_test)
# print("SVM Custom accuracy:" , custom_accuracy(y_test,y_rbf_pred2,20))
# print("SVM square: ", square(y_test,y_rbf_pred2))
# rmse = np.sqrt(mean_squared_error(y_test, y_rbf_pred2))
# print("SVM RMSE: %f" % (rmse))
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
# from sklearn.metrics import accuracy_score

data_dmatrix = xgb.DMatrix(data=X,label=y)

# xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.8, learning_rate = 0.28,max_depth = 7, n_estimators = 60)
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 1, learning_rate = 0.29, max_depth = 3, n_estimators = 70)
xg_reg.fit(X_train,y_train)

pred3 = xg_reg.predict(X_test)


print("Xgboost Custom accuracy:" ,custom_accuracy(y_test, pred3, 20))
print("Xgboost square: ", square(y_test,pred3))
rmse = np.sqrt(mean_squared_error(y_test, pred3))
print("Xgboost RMSE: %f" % (rmse))
from sklearn.ensemble import BaggingRegressor
from sklearn import tree
model = BaggingRegressor(tree.DecisionTreeRegressor(random_state=1))
model.fit(X_train, y_train)
# model.score(X_test,y_test)
y_pred2=model.predict(X_test)
print("Bagging Custom accuracy:" ,custom_accuracy(y_test, y_pred2, 20))
print("Bagging square: ", square(y_test,y_pred2))
rmse = np.sqrt(mean_squared_error(y_test, y_pred2))
print("Bagging RMSE: %f" % (rmse))
ans=(y_pred1+pred3*3+y_pred2*2)/6
print("Final Custom accuracy:" ,custom_accuracy(y_test, ans, 20))
print("Final square: ", square(y_test,ans))
rmse = np.sqrt(mean_squared_error(y_test, ans))
print("Final RMSE: %f" % (rmse))
# print(y_pred,y_test)
# Testing with a custom input
# import numpy as np
# new_prediction = lin.predict(sc.transform(np.array([[100,0,13]])))
# print("Prediction score:" , new_prediction)

# Preprocess Test data similar to Train Data
dataset1 = pd.read_excel('Data.xlsx', sheet_name='Test_Data')
dataset1['HS']=dataset1['HS'].astype(str)
dataset1['Batsman'] = 0
for i in range(0,100):
    p=dataset1['HS'][i]
    if p[-1]=='*':
        if int(p[:-1])>30:
            dataset1['Batsman'][i]=1
        dataset1['HS'][i]=p[:-1]
for i in range (0,100):
    try:
        float(dataset1['Avg'][i])
    except:
        # print("error")
        if (dataset1['Inns'][i]-dataset1['NO'][i]) == 0:
            dataset1['Avg'][i] = dataset1['2019_Runs'][i]/dataset1['Inns'][i]
        else:
            dataset1['Avg'][i]=(dataset1['2019_Runs'][i]/(dataset1['Inns'][i]-dataset1['NO'][i]))
        # print(dataset['Avg'][i])

# print(dataset['Avg'])
dataset1['Avg']=dataset1['Avg'].astype(float)
# print(dataset['Batsman'])
dataset1['HS']=dataset1['HS'].astype(int)
print(dataset1['HS'].dtype)
dataset1['BON'] = 0
for i in range (0,100):
    if dataset1['2019_Runs'][i] > 100:
        dataset1['BON'][i] = 1
print(dataset1)
X1 = dataset1.iloc[:,[2,4,6,7,10,11,13,14]].values
# Scale out input variables
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X1 = sc.fit_transform(X1)
# X_test = sc.transform(X_test)

pred5 = xg_reg.predict(X1)
pred6 = model.predict(X1)
pred7 = reg.predict(X1)
# final_pred = (pred7*1+pred6*3+pred5*2)//6
final_pred=[]
for i in range (0,100):
    final_pred.append(int((pred7[i]*1+pred6[i]*3+pred5[i]*2)/6))
print(final_pred)
output = pd.DataFrame({'Player Name ': dataset1['PLAYER'], 'Total Runs':final_pred})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")