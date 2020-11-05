# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 13:00:55 2019

@author: Dell
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 13:12:00 2019

@author: stefan
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler as sc

user = ['N5','N8','N10','N12','N13']

for user in user:
    
    userfile = pd.read_excel(f'new_data/{user}.xlsx')
    
    le = LabelEncoder()
    
    static_features = pd.read_csv('data/staticFeatures.csv',sep=';',index_col=False)
    static_features = static_features.loc[:,['holiday','weekEnd','timestamp','dayAfterHoliday','dayBeforeHoliday','dayOfYear','dayOfWeek']]
    
    weather_data = pd.read_excel('new_data/weather_data.xlsx')
    weather_data.isna().sum() 
    
#    print(weather_data['icon'].nunique(dropna=False)) # 8 values
#    print(weather_data['precipType'].nunique(dropna=False)) #3values rain snow nan(no rain no snow)
#    print(weather_data['summary'].nunique(dropna=False)) #16 values
    
    
    weather_data['precipType'] = weather_data['precipType'].fillna(value='no_rain')
    
    ## filling the rest of NaNs with interpolation
    weather_data_cleaned = weather_data.interpolate()
    
    #encoding the text data
    weather_data_cleaned['icon_encoded'] = le.fit_transform(weather_data['icon'])
    weather_data_cleaned['precipType_encoded'] = le.fit_transform(weather_data['precipType'])
    weather_data_cleaned['summary_encoded'] = le.fit_transform(weather_data['summary'])
    
    weather_data_cleaned = weather_data_cleaned.drop(['icon', 'precipType', 'summary'], axis = 1)
    
    ## data visuelization
    #N10.plot(x='stamp',y=['i1','i2','i3','pc','qc','qg','v1','v2','v3'])
    
    weather_data_cleaned_sampled = weather_data_cleaned.reset_index().set_index('time').resample('15T').mean()
    weather_data_interpolated = weather_data_cleaned_sampled.interpolate(method='linear', axis=0)
    weather_data_interpolated = weather_data_interpolated.iloc[:, weather_data_interpolated.columns != 'index']
    weather_data_interpolated = weather_data_interpolated.iloc[4:,:]
    
    static_features['timestamp'] = pd.to_datetime(static_features['timestamp'])
    static_features_sampled = static_features.reset_index().set_index('timestamp').resample('15T').mean()
    static_features_interpolated = static_features_sampled.interpolate(method='zero', axis=0)
    static_features_interpolated = static_features_interpolated.iloc[:, static_features_interpolated.columns != 'index']
    static_features_interpolated = static_features_interpolated.reset_index()
    static_features_interpolated.columns = static_features_interpolated.columns.str.replace('timestamp','time')
    static_features_interpolated = static_features_interpolated[(static_features_interpolated.time >= weather_data_interpolated.index.values[0]) & (static_features_interpolated.time <= weather_data_interpolated.index.values[len(weather_data_interpolated)-1])]
    static_features_interpolated = static_features_interpolated.set_index('time')
    
    pca =PCA(.90)
    w_d = weather_data_interpolated.iloc[:,weather_data_interpolated.columns != 'time'] # weather_data_cleaned without column 'time'
    w_d['holiday'] = static_features_interpolated['holiday']
    w_d['weekend'] = pd.Series(static_features_interpolated['weekEnd'])
    w_d['dayAfterHoliday'] = pd.Series(static_features_interpolated['dayAfterHoliday'])
    w_d['dayOfWeek'] = pd.Series(static_features_interpolated['dayOfWeek'])
    w_d['dayOfYear'] = pd.Series(static_features_interpolated['dayOfYear'])
    w_d['dayBeforeHoliday'] = pd.Series(static_features_interpolated['dayBeforeHoliday'])
#    w_d_scaled = sc().fit_transform(w_d)
#    principalComponents = pca.fit_transform(w_d_scaled)
#    principalDf = pd.DataFrame(principalComponents)
#    principalDf['time'] = weather_data_interpolated.index
    principalDf = w_d.reset_index()
    
    def equalizing(df,principalDf):
        df.columns = df.columns.str.replace('stamp','time')
        df_f = principalDf[(principalDf.time >= df['time'][0]) & (principalDf.time <= df['time'][len(df)-1])]
        missing_values_5 = pd.concat([df['time'],df_f['time']]).drop_duplicates(keep=False) 
        df_f = df_f.drop(missing_values_5.index)
        return df_f
    
    
    N_features = equalizing(userfile, principalDf)
    
    
    
    def timeincolumns(df):
        df['month'] = pd.DatetimeIndex(df['time']).month
        df['year'] = pd.DatetimeIndex(df['time']).year
        df['day'] = pd.DatetimeIndex(df['time']).day
        df['hour'] = pd.DatetimeIndex(df['time']).hour
        df['minute'] = pd.DatetimeIndex(df['time']).minute
    
    timeincolumns(N_features)
    
#    N_features = N_features.drop('time',axis=1)
    
    N_features = N_features.reset_index(drop=True)
    
    N_features.to_csv(f'features/{user}_features.csv', encoding='utf-8')
