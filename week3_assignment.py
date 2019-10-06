# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 08:33:16 2019

@author: Administrator
"""

import pandas as pd
import matplotlib.pyplot as plt

# House Sales Prices in King County
house=pd.read_csv('https://drive.google.com/uc?export=download&id=1kgJseOaDUCG-p-IoLIKbnL23XHUZPEwm')

import statsmodels.api as sm
varlist = ['bedrooms', 'bathrooms','sqft_lot', 'floors', 'waterfront', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15']

X = house[varlist]
X = sm.add_constant(X)
model = sm.OLS(house['price'],X)
result = model.fit()

print(result.summary())

# with grade, condition, view
varlist2 = ['bedrooms', 'bathrooms','sqft_lot', 'floors', 'waterfront', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15','view', 'condition', 'grade']

X = house[varlist2]
X = sm.add_constant(X)
model = sm.OLS(house['price'],X)
result = model.fit()

print(result.summary())

import numpy as np

#about data
np.sort(house['view']) #0~4
np.sort(house['condition']) #1~5
np.sort(house['grade']) #1~13

house['view'].plot.hist()
house['condition'].plot.hist()
house['grade'].plot.hist()

#compare with price

plt.scatter(house['view'],house['price'])

plt.scatter(house['condition'],house['price'])

plt.scatter(house['grade'],house['price'])
plt.scatter(house['grade'],np.log(house['price']))

#with only grade
varlist3  = ['bedrooms', 'bathrooms','sqft_lot', 'floors', 'waterfront', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15', 'grade']
X = house[varlist3]
X = sm.add_constant(X)
model = sm.OLS(np.log(house['price']),X)
result = model.fit()

print(result.summary())

# zipcode, lat, long 

np.sort(house['zipcode'])
np.sort(house['lat'])
np.sort(house['long'])

plt.boxplot(house['zipcode'])
plt.boxplot(house['lat'])
plt.boxplot(house['long'])

plt.scatter(house['zipcode'],house['price'])
plt.scatter(house['lat'],house['price'])
plt.scatter(house['long'],house['price'])
