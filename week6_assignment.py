# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 21:47:17 2019

@author: Gangmin
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import recall_score, f1_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import chi2, f_classif,mutual_info_classif

#%% For assignment 
# US Presidential elections data load 
elec=pd.read_csv('https://drive.google.com/uc?export=download&id=1fq9qDqXLiUm0un_saxAUpPsSJa05F_bV', index_col=0)
county=pd.read_csv('https://drive.google.com/uc?export=download&id=1LciKFXkb3MmpXFEHDk1Db8YFsK0liF3a') 

county.shape

data=elec.merge(county, left_on='FIPS', right_on='fips', how='left')

data['target']=(data['votes_dem_2016']>data['votes_gop_2016'])*1 

data = elec[elec['county_name']!='Alaska'].merge(county, left_on='FIPS', right_on='fips', how='left')
data_ak=elec[elec['county_name']=='Alaska'].drop_duplicates (['votes_dem_2016', 'votes_gop_2016']) 
data_ak['FIPS']=2000
data_ak=data_ak.merge(county, left_on='FIPS', right_on='fips', how='left') 
data=pd.concat((data, data_ak), axis=0).sort_values ('fips') 
data['target']=(data['votes_dem_2016']>data['votes_gop_2016'])*1

vartype = ['AFN','AGE','BPS','BZA','EDU','HSD','HSG','INC','LFE','LND','MAN','NES','POP','PST','PVY','RHI','RTN','SBO','SEX','VET','WTN']
selvars = [x for x in data.columns if x[:3] in vartype]
len(selvars)

#correlation
corr = data[['target']+selvars].corr()
corr_target = corr[['target']]
corr_target=corr_target.abs()
corr_target=corr_target.sort_values('target',ascending=False)

corr = corr[['target']]
corr=corr.drop('target')
corr['TYPE']=[x[:3] for x in corr.index]
corr['ABS_CORR']=corr['target'].abs()

mean_corr = corr.groupby(['TYPE'])['ABS_CORR'].mean()
mean_corr = mean_corr.sort_values(ascending=False)

Fscore = f_classif(data[selvars],data['target'])

Fscore = pd.DataFrame({'F':Fscore[0],'pvalue':Fscore[1]},index = selvars)
Fscore = Fscore.sort_values('F',ascending=False)

MI = mutual_info_classif(data[selvars],data['target'])
MI = pd.DataFrame({'MI':MI},index=selvars)
MI = MI.sort_values('MI',ascending = False)

var=[x for x in data.columns if 'RHI' in x or 'HSG' in x or 'AGE' in x or 'EDU' in x]
X=data[var]
y=data['target']

clf=LogisticRegression()
Cs=np.logspace(-3,3,7)

skfold=StratifiedKFold(n_splits=5, shuffle=True)
result=pd.DataFrame(columns=['C', 'Fold', 'Acc', 'Recall', 'Precision', 'F1'])
for c in Cs: 
    for pos, (train,valid) in enumerate(skfold.split(data[var], data['target'])): 
        clf.C=c 
        clf.fit(data.iloc[train][var], data.iloc[train]['target']) 
        y_pred=clf.predict(data.iloc[valid][var]) 
        result.loc[len(result)]=[c, pos+1, clf.score(data.iloc[valid][var], data.iloc[valid]['target']),recall_score(data.iloc[valid]['target'],y_pred), precision_score(data.iloc[valid]['target'], 
        y_pred), f1_score(data.iloc[valid]['target' ],y_pred)]
result.groupby('C')['Acc', 'Recall','Precision', 'F1'].mean() 

