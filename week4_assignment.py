'''
1. The purpose: Forecast sales of stores after one week
2. Split data
    ¤Training set: use samples in 2013 and 2014
    ¤Validation set: use samples in 2015
3. Select explanatory variables to predict sales
    ¤List variables used in learning
        nExplain why
4. Select learning methods to predict sales
    ¤At least two
    ¤compare the trained models
'''

#Data load
import pandas as pd

train=pd.read_csv('https://drive.google.com/uc?export=download&id=1KA7mKUmQv4PrF-qMFrH35LN6q_i56Bf1', dtype={'StateHoliday':'str'})
store=pd.read_csv('https://drive.google.com/uc?export=download&id=1_o04Vnqzo3v-MTk20MF3OMw2QFz0Fbo0')

#conver to Datetime data
train["Date"] = pd.to_datetime(train["Date"])

#slpit validation set from 2015 and train set from 2013 to 2014
val = train[train.Date.dt.year>=2015]
train = train[train.Date.dt.year<2015]

#filtering Sales
train = train[train.Sales>0]
val = val[val.Sales>0]

train.Open.unique()

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
train_new = pd.merge(train,store,how='inner',on='Store')

val_new = pd.merge(val,store,how='inner',on='Store')

#copetitionopen variables
train_new['CompetitionOpen'] = 12 * (train_new.Date.dt.year - train_new.CompetitionOpenSinceYear) + (train_new.Date.dt.month - train_new.CompetitionOpenSinceMonth)
val_new['CompetitionOpen'] = 12 * (val_new.Date.dt.year - val_new.CompetitionOpenSinceYear) + (val_new.Date.dt.month - val_new.CompetitionOpenSinceMonth)
#Promo2 variables
train_new['PromoOpen'] = 12 * (train_new.Date.dt.year - train_new.Promo2SinceYear) + (train_new.Date.dt.week - train_new.Promo2SinceWeek) / 4.0
train_new['PromoOpen']=train_new['PromoOpen'].apply(lambda x: 0 if x>20000 else x)
val_new['PromoOpen'] = 12 * (val_new.Date.dt.year - val_new.Promo2SinceYear) + (val_new.Date.dt.week - val_new.Promo2SinceWeek) / 4.0
val_new['PromoOpen']=val_new['PromoOpen'].apply(lambda x: 0 if x>20000 else x)

#Value list
varlist = ['DayOfWeek', 'Customers', 'Promo',
           'StoreType', 'Assortment',
       'CompetitionDistance', 'CompetitionOpen',
        'PromoOpen', 'PromoInterval']
#variables
X = train_new[varlist]
X_val = val_new[varlist]

#Dummy variables - Dayoweek, promo, store type, assortment, PromoInterval 
catvar=['DayOfWeek','Promo','StoreType', 'Assortment','PromoInterval']

#mean analysis
train_new.groupby('DayOfWeek')['Sales'].mean()
train_new.groupby('Promo')['Sales'].mean()
train_new.groupby('StoreType')['Sales'].mean()
train_new.groupby('Assortment')['Sales'].mean()
train_new.groupby('DayOfWeek')['Sales'].mean()

#variables
varlist = ['DayOfWeek', 'Customers', 'Promo',
           'StoreType', 'Assortment','Sales',
       'CompetitionDistance', 'CompetitionOpen',
        'PromoOpen']
X = train_new[varlist]
X_val = val_new[varlist]

X['WeekEnd'] = train_new['DayOfWeek'].apply(lambda x: 1 if x is 7 or x is 1 else 0)
X = X.drop('DayOfWeek',axis=1)
X_val['WeekEnd'] = val_new['DayOfWeek'].apply(lambda x: 1 if x is 7 or x is 1 else 0)
X_val = X_val.drop('DayOfWeek',axis=1)


X['StoreTypeB'] = train_new['StoreType'].apply(lambda x: 1 if x == 'b' else 0)
X = X.drop('StoreType',axis=1)
X_val['StoreTypeB'] = val_new['StoreType'].apply(lambda x: 1 if x == 'b' else 0)
X_val = X_val.drop('StoreType',axis=1)

catvar=['WeekEnd','Promo','StoreTypeB', 'Assortment']

for c in catvar:
    dummy = pd.get_dummies(X[c],prefix = c, drop_first=True)
    X = pd.concat((X,dummy),axis=1)
    
y = X['Sales']
X = X.drop(catvar+['Sales'],axis=1)

for c in catvar:
    dummy = pd.get_dummies(X_val[c],prefix = c, drop_first=True)
    X_val = pd.concat((X_val,dummy),axis=1)
    
y_val = X_val['Sales']
X_val = X_val.drop(catvar+['Sales'],axis=1)

#Linear regression
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
X = sm.add_constant(X)

reg = LinearRegression()
reg.fit(X,y)


model = sm.OLS(y,X)
result = model.fit()

result.summary()

#validation
X_val = sm.add_constant(X_val)
y_pred = reg.predict(X_val)
r2_score(y_val, y_pred)


#knn
from sklearn.neighbors import KNeighborsClassifier
ml = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
ml.fit(X,y)
y_pred = ml.predict(X_val)
r2_score(y_val,y_pred)


