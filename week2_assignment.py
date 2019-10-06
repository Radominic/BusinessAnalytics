# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:34:21 2019

@author: Gangmin
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets


# House Sales Prices in King County
house=pd.read_csv('https://drive.google.com/uc?export=download&id=1kgJseOaDUCG-p-IoLIKbnL23XHUZPEwm')


#todo:calculate VIF

varlist=['bedrooms', 'bathrooms', 'sqft_living','sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 
         'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15']
y=house['price']

#linear model summary
import statsmodels.api as sm
X=sm.add_constant(house[varlist])
model=sm.OLS(y, house[varlist])
result=model.fit()
print(result.summary())

from sklearn.linear_model import LinearRegression
import numpy as np

reg = LinearRegression()
VIF = []
for i in varlist:
    y=house[i]
    X=house[np.setdiff1d(varlist,[i])]
    reg.fit(X,y)
    VIF.append(1/(1-reg.score(X,y)))
df = pd.DataFrame({'value':varlist,'VIF':VIF})
print(df)

#box plot of sqft_living, sqft_above, sqft_basement
'''
plt.figure()
plt.boxplot((house['sqft_living'],house['sqft_above'],house['sqft_basement']))
plt.xticks([1, 2, 3], ['sqft_living','sqft_above','sqft_basement'])
plt.show()
'''

#hist
'''
house['sqft_living'].plot.hist(bins=20)
house['sqft_above'].plot.hist(bins=20)
house['sqft_basement'].plot.hist(bins=20)
'''
#kde
'''
house['sqft_living'].plot.kde()
house['sqft_above'].plot.kde()
house['sqft_basement'].plot.kde()
'''

#data transformation

from sklearn import preprocessing

#log
'''
reg = LinearRegression()
VIF = []
house_log = house
for i in varlist:
    house_log[i] =  preprocessing.scale(np.log(house[i]+1))
for i in varlist:
    y=house_log[i]
    X=house_log[np.setdiff1d(varlist,[i])]   
    reg.fit(X,y)
    VIF.append(1/(1-reg.score(X,y)))
df = pd.DataFrame({'value':varlist,'VIF':VIF})
print(df)

X=sm.add_constant(house[varlist])
y=house['price']
model=sm.OLS(y, house[varlist])
result=model.fit()
print(result.summary())

'''
#root
'''
reg = LinearRegression()
VIF = []
house_log = house
for i in varlist:
    house_log[i] =  preprocessing.scale(np.sqrt(house[i]+1))
for i in varlist:
    y=house_log[i]
    X=house_log[np.setdiff1d(varlist,[i])]   
    reg.fit(X,y)
    VIF.append(1/(1-reg.score(X,y)))
df = pd.DataFrame({'value':varlist,'VIF':VIF})
print(df)

X=sm.add_constant(house[varlist])
y=house['price']
model=sm.OLS(y, house[varlist])
result=model.fit()
print(result.summary())
'''
'''
#normalization
house_norm = house
for i in varlist:
     house_norm[i] = (house[i] - house[i].mean()) / (house[i].max() - house[i].min())
    
reg = LinearRegression()
VIF = []
for i in varlist:
    y=house_norm[i]
    X=house_norm[np.setdiff1d(varlist,[i])]   
    reg.fit(X,y)
    VIF.append(1/(1-reg.score(X,y)))
df = pd.DataFrame({'value':varlist,'VIF':VIF})
print(df)

X=sm.add_constant(house_norm[varlist])
y=house['price']
model=sm.OLS(y, house_norm[varlist])
result=model.fit()
print(result.summary())
'''

#removing - sqft_above, sqft_basement
'''
varlist=['bedrooms', 'bathrooms', 'sqft_living','sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 
         'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15']

reg = LinearRegression()
VIF = []
for i in varlist:
    y=house[i]
    X=house[np.setdiff1d(varlist,[i])]   
    reg.fit(X,y)
    VIF.append(1/(1-reg.score(X,y)))
df = pd.DataFrame({'value':varlist,'VIF':VIF})
print(df)

X=sm.add_constant(house[varlist])
y=house['price']
model=sm.OLS(y, house[varlist])
result=model.fit()
print(result.summary())
'''














