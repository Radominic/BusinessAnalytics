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

def vif_calculate(time_X, vif):
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(
        time_X.values, i) for i in range(time_X.shape[1])]
    vif["features"] = time_X.columns
    vif = vif.sort_values(by=['VIF Factor'],axis=0,ascending = False)
    vif = vif.reset_index(drop=True)
    
    return vif

vif = pd.DataFrame()
vif = vif_calculate(time_X, time_X)

# removelist = vif[18:]
# vif_column = removelist.features
# time_X = time_X[vif_column]

# Spot Atmospheric pressure 빼고 vif 계산 하기
newtime_X = time_X.copy()
newtime_X = newtime_X.drop('Spot atmospheric pressure_W', axis = 1)
vif = vif_calculate(newtime_X, newtime_X)

# Highest temperature 빼고 vif 계산하기
newtime_X = time_X.copy()
newtime_X = newtime_X.drop('Highest temperature_W', axis = 1)
vif = vif_calculate(newtime_X, newtime_X)

# Relative Humidity 빼고 vif 계산하기
newtime_X = time_X.copy()
newtime_X = newtime_X.drop('Relative humidity_W', axis = 1)
vif = vif_calculate(newtime_X, newtime_X)

# temperature 관련 columns 모두 제외 하고 vif 계산
temp_W = ['Average temperature_W', 'Lowest temperature_W',
       'Highest temperature_W', 'Dew point_W',
       'Vapor pressure_W', 
       'Sea-level pressure_W', 'Duration of bright sunshine_W',
       'Solar radiation quantity_W',  'Amount of cumulus_W',
       'Amount of middle and lower cloud_W', 'Average temperature of ground_W',
       'Total large evaporation_W', 'Total low evaporation_W']
newtime_X = time_X.copy()
newtime_X = newtime_X.drop(temp_W, axis = 1)
vif = vif_calculate(newtime_X, newtime_X)
# 여기에서 import weight_W 빼보기
newtime_X = newtime_X.drop('Import weight_W', axis = 1)
vif = vif_calculate(newtime_X, newtime_X)
vif



'''
# 일단 날씨 관련 칼럼만 뺄 때
while vif.iloc[0]['VIF Factor'] > 10 and vif.iloc[0]['features'] in temp_W:
    if vif.iloc[0]['features'] in temp_W:
        newtime_X = newtime_X.drop([vif.iloc[0]['features']], axis = 1)
        vif = vif.drop(0, axis = 0)
        vif = vif.reset_index(drop=True)
        newtime_X.shape
        vif = vif_calculate(newtime_X, newtime_X)
   
    
# 아래는 무역은 두고 날씨 관련만 빼고 할 때
  
while vif.iloc[2]['VIF Factor'] > 10 and vif.iloc[2]['features'] in temp_W:
    if vif.iloc[2]['features'] in temp_W:
        newtime_X = newtime_X.drop([vif.iloc[2]['features']], axis = 1)
        vif = vif.drop(2, axis = 0)
        vif = vif.reset_index(drop=True)
        newtime_X.shape
        vif = vif_calculate(newtime_X, newtime_X)
        
# Import 관련 칼럼 중 VIF가 좀 더 높은 price만 제외하고 할 때
while vif.iloc[0]['VIF Factor'] > 10:
    if vif.iloc[0]['features'] in temp_W:
        newtime_X = newtime_X.drop([vif.iloc[0]['features']], axis = 1)
        vif = vif.drop(0, axis = 0)
        vif = vif.reset_index(drop=True)
        newtime_X.shape
        vif = vif_calculate(newtime_X, newtime_X)
    elif vif.iloc[0]['features'].split()[0] == 'Import':
        newtime_X = newtime_X.drop([vif.iloc[0]['features']], axis = 1)
        vif = vif.drop(0, axis = 0)
        vif = vif.reset_index(drop=True)
        newtime_X.shape
        vif = vif_calculate(newtime_X, newtime_X)
'''

'''
# 그냥 분리

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(time_X, y, test_size = 0.2, random_state = 100)
reg = LinearRegression()
reg.fit(time_X, y)
print("train_test_split: {0:.4f}".format(reg.score(X_test, y_test)))
'''

##### Progress 4
# 1. 연도별로 묶어서 (제일 최근가격 test set으로)
# 2. 변수들끼리 묶어서 vif 체크하면서 빼기


# 243 까지가 2018 [:244]


X_trainO = newtime_X[244:] 
X_testO = newtime_X[:244] # 2018년
y_train = y[244:]
y_test = y[:244] # 2018년

columns = ['Cabbage_Price_W', 'Cabbage_Normalyear_W', 'Radish_Price_W',
       'Radish_Normalyear_W', 'Precipitation_W', 'Wind speed_W',
       'Relative humidity_W', 'Spot atmospheric pressure_W',
       'Maximum depth of snow cover_W', 'Maximum depth of new snowfall_W',
       'Import price_W']

result_matrix = []
for i in columns:
    score_list=[]
    X_train = X_trainO.copy()
    X_test = X_testO.copy()
    X_train = X_train.drop(i , axis = 1)
    X_test = X_test.drop(i , axis = 1)
    
    ## LinearRegression
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    score_list.append(reg.score(X_test, y_test))
    
    ## KneighborsRegressor
    from sklearn.neighbors import KNeighborsRegressor
    knreg = KNeighborsRegressor(n_neighbors=5)
    knreg.fit(X_train, y_train)
    score_list.append(knreg.score(X_test, y_test))
    
    
    ##  Support Vector Regressor
    from sklearn.svm import SVR
    svm_reg=SVR(kernel='poly',gamma='auto',degree=2,C=5,epsilon=0.1)
    svm_reg.fit(X_train, y_train)
    score_list.append(svm_reg.score(X_test, y_test))
    
    
    ## linearSVR
    from sklearn.svm import LinearSVR
    sv_reg=LinearSVR(max_iter=1000)
    sv_reg.fit(X_train, y_train)
    score_list.append(sv_reg.score(X_test, y_test))
    
    
    ## random forest
    from sklearn.ensemble import RandomForestRegressor
    rf_reg=RandomForestRegressor(max_depth=5)
    rf_reg.fit(X_train, y_train)
    score_list.append(rf_reg.score(X_test, y_test))
    
    '''
    ## LightGBM
    import lightgbm as lgb
    lgb_reg=lgb.LGBMRegressor(objective='regression')
    lgb_reg.fit(X_train, y_train)
    score_list.append(lgb_reg.score(X_test, y_test))
    '''
    '''
    ### XGBoost
    import xgboost as xgb
    xgb_reg=xgb.XGBRegressor()
    xgb_reg.fit(X_train, y_train)
    score_list.append(xgb_reg.score(X_test, y_test))
    '''
    
    #### Gradient Boosting Regressor
    from sklearn.ensemble import GradientBoostingRegressor
    gb_reg=GradientBoostingRegressor()
    gb_reg.fit(X_train, y_train)
    score_list.append(gb_reg.score(X_test, y_test))
    
    
    ### Decision TreeRegrssor
    from sklearn.tree import DecisionTreeRegressor
    dt_reg=DecisionTreeRegressor(max_depth=5)
    dt_reg.fit(X_train, y_train)
    score_list.append(dt_reg.score(X_test, y_test))
    
    
    #####
    method=['Linear Regression', 'KNeighbors_Regression', 'SupportVector_Regression','Linear_SV_Regression','RandomForest_Regression','GradientBoosting_Regression','DecisionTree_Regression']
    
    Result=pd.DataFrame()
    Result['method'] = method
    Result['score'] = score_list
    
    result=Result.sort_values(by='score',ascending=False)
    result_matrix.append(result)

#radish normal drop
newtime_X = newtime_X.drop('Radish_Normalyear_W',axis = 1)
X_trainO = newtime_X[244:] 
X_testO = newtime_X[:244] # 2018년
y_train = y[244:]
y_test = y[:244] # 2018년


columns = ['Cabbage_Price_W', 'Cabbage_Normalyear_W', 'Radish_Price_W',
        'Precipitation_W', 'Wind speed_W',
       'Relative humidity_W', 'Spot atmospheric pressure_W',
       'Maximum depth of snow cover_W', 'Maximum depth of new snowfall_W',
       'Import price_W']

result_matrix = []
for i in columns:
    score_list=[]
    X_train = X_trainO.copy()
    X_test = X_testO.copy()
    X_train = X_train.drop(i , axis = 1)
    X_test = X_test.drop(i , axis = 1)
    
    ## LinearRegression
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    score_list.append(reg.score(X_test, y_test))
    
    ## KneighborsRegressor
    from sklearn.neighbors import KNeighborsRegressor
    knreg = KNeighborsRegressor(n_neighbors=5)
    knreg.fit(X_train, y_train)
    score_list.append(knreg.score(X_test, y_test))
    
    
    ##  Support Vector Regressor
    from sklearn.svm import SVR
    svm_reg=SVR(kernel='poly',gamma='auto',degree=2,C=5,epsilon=0.1)
    svm_reg.fit(X_train, y_train)
    score_list.append(svm_reg.score(X_test, y_test))
    
    
    ## linearSVR
    from sklearn.svm import LinearSVR
    sv_reg=LinearSVR(max_iter=1000)
    sv_reg.fit(X_train, y_train)
    score_list.append(sv_reg.score(X_test, y_test))
    
    
    ## random forest
    from sklearn.ensemble import RandomForestRegressor
    rf_reg=RandomForestRegressor(max_depth=5)
    rf_reg.fit(X_train, y_train)
    score_list.append(rf_reg.score(X_test, y_test))
    
    '''
    ## LightGBM
    import lightgbm as lgb
    lgb_reg=lgb.LGBMRegressor(objective='regression')
    lgb_reg.fit(X_train, y_train)
    score_list.append(lgb_reg.score(X_test, y_test))
    '''
    '''
    ### XGBoost
    import xgboost as xgb
    xgb_reg=xgb.XGBRegressor()
    xgb_reg.fit(X_train, y_train)
    score_list.append(xgb_reg.score(X_test, y_test))
    '''
    
    #### Gradient Boosting Regressor
    from sklearn.ensemble import GradientBoostingRegressor
    gb_reg=GradientBoostingRegressor()
    gb_reg.fit(X_train, y_train)
    score_list.append(gb_reg.score(X_test, y_test))
    
    
    ### Decision TreeRegrssor
    from sklearn.tree import DecisionTreeRegressor
    dt_reg=DecisionTreeRegressor(max_depth=5)
    dt_reg.fit(X_train, y_train)
    score_list.append(dt_reg.score(X_test, y_test))
    
    
    #####
    method=['Linear Regression', 'KNeighbors_Regression', 'SupportVector_Regression','Linear_SV_Regression','RandomForest_Regression','GradientBoosting_Regression','DecisionTree_Regression']
    
    Result=pd.DataFrame()
    Result['method'] = method
    Result['score'] = score_list
    
    result=Result.sort_values(by='score',ascending=False)
    result_matrix.append(result)

