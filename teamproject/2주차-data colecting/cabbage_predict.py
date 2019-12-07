# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import datetime


weather = pd.read_csv("https://drive.google.com/uc?export=download&id=1vTIERY9x-lDg1beEF4ESaGuRjyMF68Bb")

radish_2015 = pd.read_csv("https://drive.google.com/uc?export=download&id=1_gxsRp4aT1ybm7I8KIowodrvPiJDuZQb")
radish_2016 = pd.read_csv("https://drive.google.com/uc?export=download&id=1BOzhm3lXKxHZLdInPT9X_q1Ly4u5Tfcv")
radish_2017 = pd.read_csv("https://drive.google.com/uc?export=download&id=1arCFioPYNxpw5MCFjWLJsLhFkFCDQCjM")
radish_2018 = pd.read_csv("https://drive.google.com/uc?export=download&id=1O1SJpmZ7CMqAQLjwza3PIvUUsCJ9pa4W")


cabbage_2015 = pd.read_csv("https://drive.google.com/uc?export=download&id=1u2t7xIopBEOIXub8V28PKehSR8w6HEbd")
cabbage_2016 = pd.read_csv("https://drive.google.com/uc?export=download&id=1jfm6QzPb8CgBDfeBkrQ8JFj96SFLzSsD")
cabbage_2017 = pd.read_csv("https://drive.google.com/uc?export=download&id=1HuGwpQ5yIgHj0CEMD4fFseZVJWJoH0QI")
cabbage_2018 = pd.read_csv("https://drive.google.com/uc?export=download&id=1_mWEXjhaG3CCNbmvtxNsNiAC5rtm45Lw")

import_2015 = pd.read_csv("https://drive.google.com/uc?export=download&id=1cLHcEqzGKnO2xVMziQkisD_db7DM_Nsx")
import_2016 = pd.read_csv("https://drive.google.com/uc?export=download&id=15LijxI-ZF1u8XydsF1D6srKLhwCbad8i")
import_2017 = pd.read_csv("https://drive.google.com/uc?export=download&id=1Q3wfdrfx7-n7zhqE1fYnDykLLq_ljevJ")
import_2018 = pd.read_csv("https://drive.google.com/uc?export=download&id=1hkDJtch3gslwSCIU1LBAJTM6jREYmCyD")
'''
our target is predicting cabbage price from other data sets.

radish_2015~2018 means radish price from 1 to 12. 
average price column means average price of seaonal product in this year.
normal year column means average price of last 30 years.

cabbage_2015~2018 means cabbage price from 1 to 12. this is average price of seaonal product.
average price column means average price of seaonal product in this year.
normal year column means average price of last 30 years.

import_2015~2018 menas month average weight and amount of cabbage.

weather is dataset from 2015~2018. 
It consider temperature, wind, rain, vapor and so on.

'''
#make date in import data
list_import = [import_2015,import_2016,import_2017,import_2018]
list_year = [2015,2016,2017,2018]
for i in range(len(list_import)):
    list_import[i]['Year'] = list_year[i]
    list_import[i]['Day'] = 1

for i in range(len(list_import)):
    list_import[i]['Date']=0
    for j in range(len(list_import[i])):
        list_import[i]['Date'][j] = datetime.date(list_import[i]['Year'][j],list_import[i]['Month'][j],list_import[i]['Day'][j])
    list_import[i] = list_import[i].drop(['Year','Month','Day'],axis=1)
    
#merge year data
cabbage = pd.concat([cabbage_2015,cabbage_2016,cabbage_2017,cabbage_2018], ignore_index=True)
radish = pd.concat([radish_2015,radish_2016,radish_2017,radish_2018], ignore_index=True)
cabbage_import =  pd.concat(list_import, ignore_index=True)

#make day on cabbage_import
cabbage_imports = pd.DataFrame()
cabbage_imports['Date'] =  cabbage['Date']
cabbage_imports['Date'] = pd.to_datetime(cabbage_imports['Date'])
cabbage_import['Date'] = pd.to_datetime(cabbage_import['Date'])

cabbage_import = pd.DataFrame(cabbage_import,columns=['Date','Import weight', 'Amount of income'])

cabbage_imports = pd.merge(cabbage_imports,cabbage_import,how='outer')
cabbage_imports = cabbage_imports.sort_values(by=['Date'], axis=0)
cabbage_imports=cabbage_imports.fillna(method ='ffill')
#조인문제 해결하

#set column name
cabbage.columns = ["Date","Cabbage_Average","Cabbage_Normalyear"]
radish.columns = ["Date","Radish_Average","Radish_Normalyear"]

#spit data by location 
list_weather = []
for i in weather.groupby(weather['Location']):
    list_weather.append(i[1])
    
#missing value
for i in range(len(list_weather)):
    print(list_weather[i].isnull().sum())

for i in range(len(list_weather)):
    list_weather[i]['Highest temperature']=list_weather[i]['Highest temperature'].fillna(method='ffill')
    list_weather[i]['Wind speed']=list_weather[i]['Wind speed'].fillna(0)
    list_weather[i]['Precipitation']=list_weather[i]['Precipitation'].fillna(0)
    list_weather[i]['Duration of bright sunshine']=list_weather[i]['Duration of bright sunshine'].fillna(0)
    list_weather[i]['Solar radiation quantity']=list_weather[i]['Solar radiation quantity'].fillna(0)
    list_weather[i]['Maximum depth of snow cover']=list_weather[i]['Maximum depth of snow cover'].fillna(0)
    list_weather[i]['Maximum depth of new snowfall']=list_weather[i]['Maximum depth of new snowfall'].fillna(0)
    list_weather[i]['Average temperature of ground']=list_weather[i]['Average temperature of ground'].fillna(method ='ffill')
    list_weather[i]['Spot atmospheric pressure']=list_weather[i]['Spot atmospheric pressure'].fillna(method ='ffill')
    list_weather[i]['Sea-level pressure']=list_weather[i]['Sea-level pressure'].fillna(method ='ffill')
    list_weather[i]['Total large evaporation']=list_weather[i]['Total large evaporation'].fillna(0)
    list_weather[i]['Toal low evaporation']=list_weather[i]['Toal low evaporation'].fillna(0)
    list_weather[i]['Relative humidity']=list_weather[i]['Relative humidity'].fillna(method ='ffill')
    list_weather[i]['Vapor pressure']=list_weather[i]['Vapor pressure'].fillna(method ='ffill')
    list_weather[i]['Toal low evaporation']=list_weather[i]['Toal low evaporation'].fillna(0)
    list_weather[i]['Toal low evaporation']=list_weather[i]['Toal low evaporation'].fillna(0)
    list_weather[i]['Average temperature']=list_weather[i]['Average temperature'].fillna(0)
    list_weather[i]['Dew point']=list_weather[i]['Dew point'].fillna(0)
    
for i in range(len(list_weather)):
    list_weather[i]['Amount of cumulus'].plot()
    list_weather[i]['Amount of middle and lower cloud'].plot()

for i in list([0,1,3,4]):
    list_weather[i]['Amount of cumulus'] = list_weather[i]['Amount of cumulus'].fillna(method ='ffill')
    list_weather[i]['Amount of middle and lower cloud'] = list_weather[i]['Amount of middle and lower cloud'].fillna(method ='ffill')
    
list_weather[2]['Amount of cumulus'] = np.where(pd.notnull(list_weather[2]['Amount of cumulus']) == True, list_weather[2]['Amount of cumulus'], list_weather[1]['Amount of cumulus'])
list_weather[2]['Amount of middle and lower cloud'] = np.where(pd.notnull(list_weather[2]['Amount of middle and lower cloud']) == True, list_weather[2]['Amount of middle and lower cloud'], list_weather[1]['Amount of middle and lower cloud'])

#average weather by location

for i in range(len(list_weather)):
    list_weather[i] = list_weather[i].reset_index(drop=True)

length = len(list_weather[0])
avg_weather = list_weather[0].copy()
columns = list_weather[0].columns
columns = ['Location',  'Average temperature', 'Lowest temperature',
       'Highest temperature', 'Precipitation', 'Wind speed', 'Dew point',
       'Relative humidity', 'Vapor pressure', 'Spot atmospheric pressure',
       'Sea-level pressure', 'Duration of bright sunshine',
       'Solar radiation quantity', 'Maximum depth of snow cover',
       'Maximum depth of new snowfall', 'Amount of cumulus',
       'Amount of middle and lower cloud', 'Average temperature of ground',
       'Total large evaporation', 'Toal low evaporation']
for i in columns:
        avg_weather[i] = list_weather[0][i].add(list_weather[1][i])
        avg_weather[i] = avg_weather[i].add(list_weather[2][i])
        avg_weather[i] = avg_weather[i].add(list_weather[3][i])
        avg_weather[i] = avg_weather[i].add(list_weather[4][i])
        avg_weather[i] = avg_weather[i].div(len(list_weather))

avg_weather = avg_weather.drop(['Location'],axis=1)

#merge data
X = pd.merge(cabbage, radish, how='left')
X = pd.merge(X,avg_weather,how='left')
X['Date'] = pd.to_datetime(X['Date'])
X = pd.merge(X,cabbage_imports,how='left')

