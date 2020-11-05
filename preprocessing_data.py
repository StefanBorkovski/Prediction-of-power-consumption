# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 10:05:35 2018

@author: stefan
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime


staticFeatures = pd.read_csv('data/staticFeatures.csv', delimiter = ';')
descpription = pd.read_csv('data/description.csv', delimiter = ';')

N5 = pd.read_json('data/N5.json')
N8 = pd.read_json('data/N8.json')
N10 = pd.read_json('data/N10.json')
N12 = pd.read_json('data/N12.json')
N13 = pd.read_json('data/N13.json')

weather_data = pd.read_json('data/weather.json', lines = True)
    
N5['stamp'] = pd.to_datetime(N5['stamp'], unit = 's')
N5['stamp_db'] = pd.to_datetime(N5['stamp_db'], unit = 's')

N8['stamp'] = pd.to_datetime(N8['stamp'], unit = 's')
N8['stamp_db'] = pd.to_datetime(N8['stamp_db'], unit = 's')

N10['stamp'] = pd.to_datetime(N10['stamp'], unit = 's')
N10['stamp_db'] = pd.to_datetime(N10['stamp_db'], unit = 's')

N12['stamp'] = pd.to_datetime(N12['stamp'], unit = 's')
N12['stamp_db'] = pd.to_datetime(N12['stamp_db'], unit = 's')

N13['stamp'] = pd.to_datetime(N13['stamp'], unit = 's')
N13['stamp_db'] = pd.to_datetime(N13['stamp_db'], unit = 's')

hourly_data = weather_data['hourly']
#hour1 = hourly_data.map
#hour2 = pd.DataFrame(hourly_data['data'])

weather_hourly = pd.concat([pd.DataFrame(hourly_data[i]['data'][0], index = [i]) for i in range(hourly_data.shape[0])], sort = True)
weather_hourly['time'] = pd.to_datetime(weather_hourly['time'], unit = 's')

weather_hourly.isna().sum() 
weather_hourly = weather_hourly.drop(['precipIntensity','precipProbability','windGust', 'ozone'], axis = 1)


weather_hourly.to_excel('new_data/weather_data.xlsx', encoding='utf-8')
N5.to_excel('new_data/N5.xlsx', encoding='utf-8')
N8.to_excel('new_data/N8.xlsx', encoding='utf-8')
N10.to_excel('new_data/N10.xlsx', encoding='utf-8')
N12.to_excel('new_data/N12.xlsx', encoding='utf-8')
N13.to_excel('new_data/N13.xlsx', encoding='utf-8')