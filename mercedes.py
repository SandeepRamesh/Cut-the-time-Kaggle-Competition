# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 13:05:05 2017

@author: Sandeep Ramesh
"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import xgboost as xgb
mercedes_train = pd.read_csv('train.csv',index_col='ID')
mercedes_train=mercedes_train.sort_index()
mercedes_test=pd.read_csv('test.csv',index_col='ID')

#getting the info on dataset
mercedes_train.info()
mercedes_test.info()
mercedes_train.describe()
mercedes_train=mercedes_train.drop_duplicates()
#mercedes_test=mercedes_test.drop_duplicates()

#getting the plot of predictor variable
sns.distplot(mercedes_train.y)
plt.scatter(range(mercedes_train.shape[0]),np.sort(mercedes_train.y.values))
plt.xlabel('ID')
plt.ylabel('Y values')
plt.show()

#getting the top missing values
mercedes_train.dtypes.value_counts()
mercedes_train.isnull().sum().sort_values(ascending=False).head()


#Getting the box plots of all categorical variables 
for col in mercedes_train.columns:
    if (mercedes_train[col].dtypes=='object'):
        sns.boxplot(x=col,y=mercedes_train.y,data=mercedes_train)
        plt.xticks(rotation = 70)
        plt.show()

#x4 might not be significant after seeing the plots
#for Test purposes
X=mercedes_train.drop('y',1)
X=pd.get_dummies(X,drop_first=True)
y=mercedes_train['y']


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#Getting the column differences between train and test
set(x_train.columns.tolist()) - set(x_test.columns.tolist())
#Getting the column differences between test and train
set(x_test.columns.tolist()) - set(x_train.columns.tolist())
'''
#results of above differences
x_test=x_test.drop(['X0_ae',
 'X0_ag',
 'X0_an',
 'X0_av',
 'X0_bb',
 'X0_p',
 'X2_ab',
 'X2_ad',
 'X2_aj',
 'X2_ax',
 'X2_u',
 'X2_w',
 'X5_aa',
 'X5_b',
 'X5_t',
 'X5_z'],1)
x_train=x_train.drop(['X0_aa',
 'X0_ab',
 'X0_ac',
 'X0_q',
 'X2_aa',
 'X2_ar',
 'X2_c',
 'X2_l',
 'X2_o',
 'X5_u'],1)
 '''   
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFECV
regressor = RandomForestRegressor(n_estimators=200, max_depth=10,
                                  min_samples_leaf=4, max_features=0.2, 
                                  n_jobs=-1, random_state=0)
rfecv = RFECV(estimator=regressor, step=1, cv=10)
rfecv.fit(x_train, y_train)
print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

#new dataframe with features and their rankings
ranking=pd.DataFrame({'Features': x_train.columns})
ranking['rank'] = rfecv.ranking_
ranking.sort_values('rank',inplace=True)
ranking=ranking.reset_index()
ranking.to_csv('Ranking.csv',index=False)

#creating a df with 23 best rfecv features
ranklist=ranking.head(23)
ranklist.Features.tolist()
newdf_X=x_train[['X127','X54','X0_az','X261','X263','X136','X29','X76','X238','X234',
                 'X316','X315','X328','X232','X115','X118','X279','X314','X119','X311',
                 'X313','X275','X348']]
newdf_X=newdf_X.reset_index()
newdf_X=newdf_X.drop(['ID'],1)
newdf_Y=y_train.values.ravel()
newdf_Y=newdf_Y.reset_index()
newdf_Y=newdf_Y.drop(['ID'],1)
#splitting train into train test split for r2 values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(newdf_X, newdf_Y, test_size = 0.2, random_state = 0)
from sklearn.metrics import r2_score

from xgboost import XGBRegressor
regressor = XGBRegressor(n_estimators=200,learning_rate=0.05)
regressor=regressor.fit(x_train, y_train)

# Predicting the Test set results
predictions1 = regressor.predict(x_test)
print ("R^2 is: \n", r2_score(y_test,predictions1))
print ('RMSE is: \n', mean_squared_error(y_test, predictions1))

from sklearn.ensemble import RandomForestRegressor
regressor1 = RandomForestRegressor(n_estimators = 200, random_state = 42)
regressor1.fit(x_train, y_train)

# Predicting the Test set results
predictions2 = regressor1.predict(x_test)
print ("R^2 is: \n", r2_score(y_test,predictions2))
print ('RMSE is: \n', mean_squared_error(y_test, predictions2))

from sklearn.ensemble import ExtraTreesRegressor
regressor2=ExtraTreesRegressor(n_estimators=200,random_state=42)
regressor2.fit(x_train,y_train)
predictions3 = regressor2.predict(x_test)
print ("R^2 is: \n", r2_score(y_test,predictions3))
print ('RMSE is: \n', mean_squared_error(y_test, predictions3))


from sklearn.linear_model import ElasticNet
clf2 = ElasticNet(alpha=0.014, l1_ratio=0.5)
clf2.fit(x_train, y_train)
elas_preds = (clf2.predict(x_test))
print("r2 score",r2_score(y_test,elas_preds))
print ('RMSE is: \n', mean_squared_error(y_test, elas_preds))

import lightgbm as lgb
gbm = lgb.LGBMRegressor(objective='regression',learning_rate=0.05,max_depth=3,n_estimators=150)
gbm=gbm.fit(x_train, y_train)
predictions4 = (gbm.predict(x_test))
print ("R^2 is: \n", r2_score(y_test,predictions4))
print ('RMSE is: \n', mean_squared_error(y_test, predictions4))

from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(n_estimators=200,max_depth=3,learning_rate=0.05)
gbr=gbr.fit(x_train, y_train)
predictions5 = (gbr.predict(x_test))
print ("R^2 is: \n", r2_score(y_test,predictions5))
print ('RMSE is: \n', mean_squared_error(y_test, predictions5))


#For actual prediciton
x_train=mercedes_train.drop('y',1)
x_train=pd.get_dummies(x_train,drop_first=True)
y_train=mercedes_train['y']
x_test=mercedes_test
x_test=pd.get_dummies(x_test,drop_first=True)

#final predictions

newdf_xtrain=x_train[['X127','X54','X0_az','X261','X263','X136','X29','X76','X238','X234',
                      'X316','X315','X328','X232','X115','X118','X279','X314','X119','X311',
                      'X313','X275','X348']]
newdf_xtrain=newdf_xtrain.reset_index()
newdf_xtrain=newdf_xtrain.drop(['ID'],1)
newdf_ytrain=y_train.values.ravel()
newdf_Y=newdf_Y.reset_index()
newdf_Y=newdf_Y.drop(['ID'],1)
newdf_xtest=x_test[['X127','X54','X0_az','X261','X263','X136','X29','X76','X238','X234',
                      'X316','X315','X328','X232','X115','X118','X279','X314','X119','X311',
                      'X313','X275','X348']]

/*Model selection and validation hidden for confidentiality purposes*/


#average of xgb, Light GBM and Gradient Boosting Regressor
final=(predictions1+predictions4+predictions5)/3


results=pd.DataFrame({"ID":newdf_xtest.index,"y":final})
results.to_csv('Final_Output.csv',index=False)
