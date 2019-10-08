# -*- coding: utf-8 -*-
"""
1. Update models
    Add new variables
     -Include variables generated from Promo2 and Promo2Interval
     -Include variable(s) from CompetitionDistance
     -Include variable(s) from CompetitionOpenSince[Month/Year]
     -Include new variable(s) from historical sales or number of customers
    Do not use ‘Store’ variable as an input variable
2. Split data
    Training set: use samples in 2013 and 2014
    Validation set: use samples in 2015
3. Create models
    Compare linear, ridge and Lasso regression methods
     -For ridge and Lasso, test several 𝜆𝜆values to find the best setting.
4. Evaluate performance of models
    Use 5-fold cross validation on training set to find the best parameter setting
    Calculate R^2
5. Summarize procedures and results
    Describe explanatory variables
    Explain the results from different algorithms
"""


#Data load
import pandas as pd

train=pd.read_csv('https://drive.google.com/uc?export=download&id=1KA7mKUmQv4PrF-qMFrH35LN6q_i56Bf1', dtype={'StateHoliday':'str'})
store=pd.read_csv('https://drive.google.com/uc?export=download&id=1_o04Vnqzo3v-MTk20MF3OMw2QFz0Fbo0')

#conver to Datetime data
train["Date"] = pd.to_datetime(train["Date"])

#filtering Sales
trainO = train[train.Sales>0]

trainO.Open.unique()

#fill nan
store.isnull().sum()

from scipy import stats

store['CompetitionDistance'].plot()

store['CompetitionDistance'].mean()
store['CompetitionDistance'].median()
stats.trim_mean(store['CompetitionDistance'],0.1)

store['CompetitionDistance'].fillna(stats.trim_mean(store['CompetitionDistance'],0.1),inplace=True)


store['CompetitionOpenSinceMonth'].plot()
store['CompetitionOpenSinceMonth'].median()

store['CompetitionOpenSinceMonth'].fillna(store['CompetitionOpenSinceMonth'].median(),inplace=True)

store['CompetitionOpenSinceYear'].plot()
store['CompetitionOpenSinceYear'].median()

store['CompetitionOpenSinceYear'].fillna(store['CompetitionOpenSinceYear'].median(),inplace=True)

store['Promo2SinceWeek'].fillna(0,inplace=True)
store['Promo2SinceYear'].fillna(0,inplace=True)
store['PromoInterval'].fillna(0,inplace=True)


#new train set
train_new = pd.merge(trainO,store,how='inner',on='Store')

#copetitionopen variables
train_new['CompetitionOpen'] = 12 * (train_new.Date.dt.year - train_new.CompetitionOpenSinceYear) + (train_new.Date.dt.month - train_new.CompetitionOpenSinceMonth)
#Promo2 variables
train_new['PromoOpen'] = 12 * (train_new.Date.dt.year - train_new.Promo2SinceYear) + (train_new.Date.dt.week - train_new.Promo2SinceWeek) / 4.0
train_new['PromoOpen']=train_new['PromoOpen'].apply(lambda x: 0 if x>20000 else x)

#Value list
varlist = ['DayOfWeek', 'Promo',
           'StoreType', 'Assortment',
       'CompetitionDistance', 'CompetitionOpen',
        'PromoOpen']
#variables
X = train_new[varlist]

#Dummy variables - Dayoweek, promo, store type, assortment, PromoInterval 
catvar=['DayOfWeek','Promo','StoreType', 'Assortment','PromoInterval','SchoolHoliday','StateHoliday']

#mean analysis
train_new.groupby('DayOfWeek')['Sales'].mean()
train_new.groupby('Promo')['Sales'].mean()
train_new.groupby('StoreType')['Sales'].mean()
train_new.groupby('Assortment')['Sales'].mean()
train_new.groupby('PromoInterval')['Sales'].mean()
train_new.groupby('SchoolHoliday')['Sales'].mean()
train_new.groupby('StateHoliday')['Sales'].mean()
#variables
varlist = ['Store','Date','DayOfWeek',  'Promo',
           'StoreType', 'Assortment','Sales',
       'CompetitionDistance', 'CompetitionOpen',
        'PromoOpen','StateHoliday','Open','Customers']
X = train_new[varlist]

X['WeekEnd'] = train_new['DayOfWeek'].apply(lambda x: 1 if x is 7 or x is 1 else 0)
X = X.drop('DayOfWeek',axis=1)


X['StoreTypeB'] = train_new['StoreType'].apply(lambda x: 1 if x == 'b' else 0)
X = X.drop('StoreType',axis=1)

catvar=['WeekEnd','Promo','StoreTypeB', 'Assortment','StateHoliday']

for c in catvar:
    dummy = pd.get_dummies(X[c],prefix = c, drop_first=True)
    X = pd.concat((X,dummy),axis=1)
    


#historical variables
train_new.groupby(train_new['Date'].dt.month)['Sales'].mean()
import numpy as np

def moving_average(data,col,window,lag=0):
    temp1 = pd.DataFrame()
    temp2 = pd.DataFrame()
    for i in range(1+lag,window+1+lag):
        temp1['%s%d'%(col,i)]=data.groupby('Store')[col].shift(i).values
        temp2['Open%d'%(i)]=data.groupby('Store')['Open'].shift(i).values
    return (np.sum(temp1.values*temp2.values,1)/temp2.sum(1)).values


sel_train = pd.DataFrame()
sel_train['Sales1W']=moving_average(X,'Sales',7,7)

sel_train['Sales2W']=moving_average(X,'Sales',14,7)
sel_train['Sales3W']=moving_average(X,'Sales',21,7)
sel_train['Sales4W']=moving_average(X,'Sales',28,7)
sel_train['Sales1W_4W_diff']=sel_train['Sales1W']-sel_train['Sales4W']
sel_train['Sales1W_4W_ratio']=sel_train['Sales1W']/sel_train['Sales4W']


sel_train['Customers1W']=moving_average(X,'Customers',7,7)

sel_train['Customers2W']=moving_average(X,'Customers',14,7)
sel_train['Customers3W']=moving_average(X,'Customers',21,7)
sel_train['Customers4W']=moving_average(X,'Customers',28,7)
sel_train['Customers1W_4W_diff'] = sel_train['Customers1W']-sel_train['Customers4W']
sel_train['Customers1W_4W_ratio'] = sel_train['Customers1W']/sel_train['Customers4W']


sel_train['Sales'] = X['Sales']
sel_train = sel_train[sel_train.Sales>0]

X['Sales1W_4W_ratio'] = sel_train['Sales1W_4W_ratio']
X['Customers1W_4W_ratio'] = sel_train['Customers1W_4W_ratio']


X['Sales1W']=sel_train['Sales1W']

X['Sales2W']=sel_train['Sales2W']
X['Sales3W']=sel_train['Sales3W']
X['Sales4W']=sel_train['Sales4W']
X['Sales1W_4W_diff']=sel_train['Sales1W_4W_diff']

X['Customers1W']=sel_train['Customers1W']

X['Customers2W']=sel_train['Customers2W']
X['Customers3W']=sel_train['Customers3W']
X['Customers4W']=sel_train['Customers4W']
X['Customers1W_4W_diff'] =sel_train['Customers1W_4W_diff'] 


X = X.dropna()
X = X.drop(['Store']+catvar,axis=1)

# target variables

y = X['Sales']
X = X.drop(['Sales'],axis=1)
X = X.drop(['Customers'],axis =1)
#data normalization 
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()


X['CompetitionDistance']= min_max_scaler.fit_transform(X[['CompetitionDistance']])
X['CompetitionOpen']= min_max_scaler.fit_transform(X[['CompetitionOpen']])
X['PromoOpen']= min_max_scaler.fit_transform(X[['PromoOpen']])

X['Sales1W']= min_max_scaler.fit_transform(X[['Sales1W']])

X['Sales2W']= min_max_scaler.fit_transform(X[['Sales2W']])
X['Sales3W']= min_max_scaler.fit_transform(X[['Sales3W']])
X['Sales4W']= min_max_scaler.fit_transform(X[['Sales4W']])
X['Sales1W_4W_diff']= min_max_scaler.fit_transform(X[['Sales1W_4W_diff']])

X['Customers1W']= min_max_scaler.fit_transform(X[['Customers1W']])

X['Customers2W']= min_max_scaler.fit_transform(X[['Customers2W']])
X['Customers3W']= min_max_scaler.fit_transform(X[['Customers3W']])
X['Customers4W']= min_max_scaler.fit_transform(X[['Customers4W']])
X['Customers1W_4W_diff']= min_max_scaler.fit_transform(X[['Customers1W_4W_diff']])



#seasonality
X['month'] = X['Date'].dt.month
X['season'] = X['month'].apply(lambda x: 'spring' if x == 3 or x==4 or x== 5 else ('summer' if x==6 or x==7 or x== 8 else('fall' if x== 9 or x== 10 or x== 11 else 'winter' )))
X = X.drop(['Date','month'],axis=1)

dummy = pd.get_dummies(X['season'],prefix = 'season', drop_first=True)
X = pd.concat((X,dummy),axis=1)
X = X.drop(['season'],axis=1)

#reindexing
X = X.reset_index()
y = y.reset_index()

X =X.drop(['index'],axis=1)
y =y.drop(['index'],axis=1)


#kfold validation
from sklearn.model_selection import KFold
kf = KFold(n_splits = 5,shuffle=True, random_state = 1)

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score

#linear regression
for train_index, test_index in kf.split(X):
    train_x = X.iloc[train_index]
    train_y = y.iloc[train_index]
    test_x = X.iloc[test_index]
    test_y = y.iloc[test_index]
    
    reg = LinearRegression()
    reg.fit(train_x,train_y)
    predict_y = reg.predict(test_x)
    print(r2_score(test_y,predict_y))
    print(reg.coef_)
    
#Lasso regression
    
for train_index, test_index in kf.split(X):
    train_x = X.iloc[train_index]
    train_y = y.iloc[train_index]
    test_x = X.iloc[test_index]
    test_y = y.iloc[test_index]
    
    lasso = Lasso(alpha = 0.01)
    lasso.fit(train_x,train_y)
    predict_y = lasso.predict(test_x)
    print(r2_score(test_y,predict_y))
    print(lasso.coef_)
    
#Ridge regression

for train_index, test_index in kf.split(X):
    train_x = X.iloc[train_index]
    train_y = y.iloc[train_index]
    test_x = X.iloc[test_index]
    test_y = y.iloc[test_index]
    
    ridge = Ridge(alpha = 0.01)
    ridge.fit(train_x,train_y)
    predict_y = ridge.predict(test_x)
    print(r2_score(test_y,predict_y))
    print(ridge.coef_)




