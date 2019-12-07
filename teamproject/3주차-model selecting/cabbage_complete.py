# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
X=pd.read_csv("https://drive.google.com/uc?export=download&id=1qZNuUw38sW3Wu4ndRcOp4EU5OVqjUKER",index_col=[0])

#정렬
X = X.sort_values(by=['Date'],axis=0,ascending=False)
X = X.reset_index(drop=True)

#날짜변수만들기
X['Date'] = pd.to_datetime(X['Date'])

X['Year'] = X['Date'].dt.year
X['Month'] = X['Date'].dt.month
X['Day'] = X['Date'].dt.day
X['DoW']=X['Date'].dt.dayofweek

new_X = X[['Date','Year','Month','Day','DoW']]


#타임변수만들기
def moving_average(data,col,window,lag=0):
    temp = pd.DataFrame()
    for i in range(1+lag,window+1+lag):
        temp['%s%d'%(col,i)]=data[col].shift(i).values
    return np.sum(temp.values,1)/window

col_list = ['Cabbage_Price', 'Cabbage_Normalyear', 'Radish_Price',
       'Radish_Normalyear', 'Average temperature', 'Lowest temperature',
       'Highest temperature', 'Precipitation', 'Wind speed', 'Dew point',
       'Relative humidity', 'Vapor pressure', 'Spot atmospheric pressure',
       'Sea-level pressure', 'Duration of bright sunshine',
       'Solar radiation quantity', 'Maximum depth of snow cover',
       'Maximum depth of new snowfall', 'Amount of cumulus',
       'Amount of middle and lower cloud', 'Average temperature of ground',
       'Total large evaporation', 'Total low evaporation', 'Import weight',
       'Import price']


time_X = pd.DataFrame()
for i in col_list:
    time_X[i+'_W'] = moving_average(X,i,7,7)

    
    
time_X['Date'] = X['Date']
time_X['price'] = X['Cabbage_Price']

time_X = time_X[14:]#원래 57
time_X = time_X.reset_index(drop=True)
t=time_X.copy()
#정규화
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
for i in col_list:
    time_X[i+'_W'] = min_max_scaler.fit_transform(time_X[[i+'_W']])
    
time_X['price'] =  min_max_scaler.fit_transform(time_X[['price']])
y = time_X['price']
time_X = time_X.drop(['Date','price'],axis=1)



# vif
from statsmodels.stats.outliers_influence import variance_inflation_factor

def vif_calculate(vif):
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(
        time_X.values, i) for i in range(time_X.shape[1])]
    vif["features"] = time_X.columns
    vif = vif.sort_values(by=['VIF Factor'],axis=0,ascending = False)
    vif = vif.reset_index(drop=True)
    
    return vif

vif = pd.DataFrame()
vif = vif_calculate(time_X)

removelist = vif[18:]
vif_column = removelist.features


time_X = time_X[vif_column]

#회귀그냥 해봄

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
time_X = sm.add_constant(time_X)

reg = LinearRegression()
reg.fit(time_X,y)


model = sm.OLS(y,time_X)
result = model.fit()

result.summary()

# 그냥 분리
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(time_X, y, test_size = 0.2, random_state = 100)
reg = LinearRegression()
reg.fit(time_X, y)
print("train_test_split: {0:.4f}".format(reg.score(X_test, y_test)))

# 그냥 K-Fold
from sklearn.model_selection import cross_val_score, KFold

kfold_scores = []
for i in range(3, 20):
    kf = KFold(n_splits = i, shuffle = True, random_state = 1)
    reg = LinearRegression()
    scores = cross_val_score(reg, time_X, y, cv = kf)
    kfold_scores.append(scores.mean())
max(kfold_scores)


#del time_X['price']
#del time_X['Date']
# k = 10번, random state = 1
score_list=[]
## LinearRegression
kf = KFold(n_splits = 4, shuffle = True, random_state = 100)
reg = LinearRegression()
scores = cross_val_score(reg, time_X, y, cv = kf)
print("LinearRegrssion: {0:.4f}".format(scores.mean()))
score_list.append(scores.mean())

## KneighborsRegressor
from sklearn.neighbors import KNeighborsRegressor

knreg = KNeighborsRegressor(n_neighbors=5)
knscores = cross_val_score(knreg, time_X, y, cv = kf)
print("KNeighborsRegressor: {0:.4f}".format(knscores.mean()))
score_list.append(knscores.mean())

##  Support Vector Regressor
from sklearn.svm import SVR

svm_reg=SVR(kernel='poly',gamma='auto',degree=2,C=5,epsilon=0.1)
svm_score=cross_val_score(svm_reg,time_X,y,cv=kf)
print("Support Vector Regressor: {0:.4f}".format(svm_score.mean()))
score_list.append(svm_score.mean())



## linearSVR

from sklearn.svm import LinearSVR

sv_reg=LinearSVR(max_iter=1000)
sv_score=cross_val_score(sv_reg,time_X,y,cv=kf)
print("LinearSVR : {0:.4f}".format(sv_score.mean()))
score_list.append(sv_score.mean())

## random forest

from sklearn.ensemble import RandomForestRegressor

rf_reg=RandomForestRegressor(max_depth=5)
rf_score=cross_val_score(rf_reg,time_X,y,cv=kf)
print("Random Forest Regressor : {0:.4f}".format(rf_score.mean()))
score_list.append(rf_score.mean())

## LightGBM

import lightgbm as lgb

lgb_reg=lgb.LGBMRegressor(objective='regression')
lgb_score=cross_val_score(lgb_reg,time_X,y,cv=kf)
print("Light GBM Regression : {0:.4f}".format(lgb_score.mean()))
score_list.append(lgb_score.mean())
'''
### XGBoost

import xgboost as xgb

xgb_reg=xgb.XGBRegressor()
xgb_score=cross_val_score(xgb_reg,time_X,y,cv=kf)
print("XGBoost Regression : {0:.4f}".format(xgb_score.mean()))
score_list.append(xgb_score.mean())
'''
#### Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor

gb_reg=GradientBoostingRegressor()
gb_score=cross_val_score(gb_reg,time_X,y,cv=kf)
print("Gradient Boosting Regression : {0:.4f}".format(gb_score.mean()))
score_list.append(gb_score.mean())

### Decision Tree Regrssor

from sklearn.tree import DecisionTreeRegressor

dt_reg=DecisionTreeRegressor(max_depth=5)
dt_score=cross_val_score(dt_reg,time_X,y,cv=kf)
print("Decision Tree Regression : {0:.4f}".format(dt_score.mean()))
score_list.append(dt_score.mean())

#####

method=['Linear Regression', 'KNeighbors_Regression', 'SupportVector_Regression','Linear_SV_Regression','RandomForest_Regression','Light_GBM_Regression','GradientBoosting_Regression','DecisionTree_Regression']

Result=pd.DataFrame()
Result['method']=method
Result['score']=score_list

result=Result.sort_values(by='score',ascending=False)
result
