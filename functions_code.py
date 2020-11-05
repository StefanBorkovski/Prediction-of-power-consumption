# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 16:00:36 2019

@author: Stefan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pandas.plotting import autocorrelation_plot
from math import sqrt
import datetime
import os

####    INNER FUNCTIONS     ####    (not to be called in main)

#create 3d inputs for lstm    validation and train samples (default validation size 3 weeks)
def input_for_LSTM(features, output, validation_size, test_size):
    features = features.copy(deep=True)
    output = output.copy(deep=True)
    
    scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
    features = scaler.fit_transform(features)
    features = pd.DataFrame(features)
    X_train = features.iloc[:-validation_size,:]
    X_val = features.iloc[-validation_size:-test_size,:]
    y_train = output['pc'][:-validation_size]
    y_val = output['pc'][-validation_size:-test_size]
    X_train = X_train.as_matrix(columns = None)
    X_train = X_train.reshape(X_train.shape[0:][0],1,X_train.shape[1:][0])
    X_val = X_val.as_matrix(columns = None)
    X_val = X_val.reshape(X_val.shape[0:][0],1,X_val.shape[1:][0])
  
    return X_train, X_val, y_train, y_val

#create 3d test samples for the lstm
def test_for_LSTM(features, output, test_size):
    features = features.copy(deep=True)
    output = output.copy(deep=True)  
    scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
    features = scaler.fit_transform(features)
    features = pd.DataFrame(features)
    X_test = features.iloc[-test_size:,:]
    y_test = output['pc'][-test_size:]
    print(type(X_test))
    X_test = X_test.values #.as_matrix(columns = None)
    X_test = X_test.reshape(X_test.shape[0:][0],1,X_test.shape[1:][0])
    
    return X_test, y_test
    
# sqrt for mean squared error    
def evaluation_rmse(y_test, y_pred):
    result = mean_squared_error(y_test, y_pred)
    result = sqrt(result)
    return result


#spilt the data in train and test samples for the random tree regresssor
def split_data(features, output, num_samples):
    X_train = features.iloc[:-num_samples, :]
    X_test = features.iloc[-num_samples:, :]
    y_train = output['pc'].iloc[:-num_samples]
    y_test = output['pc'].iloc[-num_samples:]
    
    return X_train, X_test, y_train, y_test
    
#make prediction (default test size 1 week)
def model_predict(model, X_test, y_test):
    y_predict = model.predict(X_test)
    rmse = evaluation_rmse(y_test, y_predict)
    return y_predict, int(rmse)

        
#draw the plot with real, predicted power and their error on the same graph
def drawPlot(y_test, y_pred, score, output, name, num):
    error_arr = []
    for i in range(len(y_pred)):
        error = abs(y_test.iloc[i] - y_pred[i])
        error_arr.append(error)
    
    plt.figure()
    plt.plot(output['stamp'][-num:],error_arr, label = 'error')
    plt.plot(output['stamp'][-num:],y_test, label = 'test')
    plt.plot(output['stamp'][-num:],y_pred, label = 'prediction')
    plt.legend()
    plt.suptitle('error over time {} acc = {}'.format(name, score))
    plt.show()

####    AVALIVABLE FUNCTIONS  FOR MAIN   ####    
    
#which features has direct correlation with the output
def feature_correlation(df, output):   
    print('feature correlation')
    df = df.copy(deep=True)
    df['pc'] = output['pc']
    corr_matrix = df.corr()
    print(corr_matrix["pc"].sort_values(ascending=False))
  
    
#check the correlation betwwen the current and the past values of 'pc'
def plotcorrelation(df, output, num_of_samples): 
    df = df.copy(deep=True)
    df['pc'] = output['pc']
    autocorrelation_plot(df.pc[:num_of_samples])
    

#only possile for N10, N12, N13
def add_past_values_currents(features, output, user, time_frame = 672):
    if user != 'N5' and user != 'N8':
        features = features[time_frame:]
        i1 = output.i1[:-time_frame]
        i2 = output.i2[:-time_frame]
        i3 = output.i3[:-time_frame]
        features = features.reset_index(drop=True)
        i1 = i1.reset_index(drop=True)
        i2 = i2.reset_index(drop=True)
        i3 = i3.reset_index(drop=True)
        features['i1_past'] = i1
        features['i2_past'] = i2
        features['i3_past'] = i3
        output = output[time_frame:]
        output = output.reset_index(drop=True)
        
    return features, output

#use past values of power consumed as features to predict the future ones
def add_past_values_power(features, output, time_frame = 672):
    features = features[time_frame:]
    pc = output.pc[:-time_frame]
    features = features.reset_index(drop=True)
    pc = pc.reset_index(drop=True)
    if time_frame == 672:
        features['pc_w'] = pc
    else:
        features['pc_past'] = pc
    output = output[time_frame:]
    output = output.reset_index(drop=True)
    
    return features, output


#feature importance using random fores regressor
def feature_importance_RF(features, output):
    print('feature importance using RF')
    rf = RandomForestRegressor(n_estimators = 100)
    rf.fit(features, output['pc'])
    feature_importances = pd.DataFrame(rf.feature_importances_,
                                       index = features.columns,
                                       columns=['importance']).sort_values('importance',ascending=False)
    print(feature_importances)
    
#feature importance using KBest
def feature_importance_KBest(features, output):
    features = features.copy(deep=True)
    output = output.copy(deep=True)
    
    test = SelectKBest(score_func=chi2, k = 6)
    fit = test.fit(features, output['pc'])
    np.set_printoptions(precision=3)
    feature_importances = pd.DataFrame(fit.scores_,
                                       index = features.columns,
                                       columns=['importance']).sort_values('importance',ascending=False)
    print(feature_importances)

#create lstm model
#(validation size 3 weeks, test size 1 week, draw plot, return model, early stopping patience 10 epochs)
def create_model_N5(features, output, EPOCHS, num_test):
    features = features.copy(deep=True)
    output = output.copy(deep=True)
    
    X_train, X_val, y_train, y_val = input_for_LSTM(features, output, 4*672, num_test)
    X_test, y_test = test_for_LSTM(features, output, num_test)
    
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1:]), activation='relu', return_sequences=True))
    model.add(LSTM(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))
    
    callbacks = [EarlyStopping(monitor='val_loss', patience = 10), 
             ModelCheckpoint(filepath = f'checkpoint/best_model N5.hdf5')]
    
    model.compile(
        loss='mae',  # dont change it
        optimizer='adadelta',
        metrics=['accuracy'])
    model.fit(X_train,
              y_train,
              callbacks = callbacks,
              epochs=EPOCHS,
              validation_data=(X_val, y_val))
    
    y_pred, rmse = model_predict(model, X_test, y_test)
    drawPlot(y_test, y_pred, rmse, output, 'KERAS', num_test)
    
    return model

def create_model_N12(features, output, EPOCHS, num_test):
    features = features.copy(deep=True)
    output = output.copy(deep=True)
    
    X_train, X_val, y_train, y_val = input_for_LSTM(features, output, 4*672, num_test)
    X_test, y_test = test_for_LSTM(features, output, num_test)
    print(X_train.shape[0:])
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1:]), activation='relu', return_sequences=True))
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=True))
    model.add(LSTM(16, activation='relu'))
    model.add(Dense(1, activation='linear'))
    
    callbacks = [EarlyStopping(monitor='val_loss', patience = 10), 
             ModelCheckpoint(filepath = f'checkpoint/best_model N12.hdf5')]
    model.compile(
        loss='mae', # don't change it
        optimizer='adadelta',
        metrics=['accuracy'])
    model.fit(X_train,
              y_train,
              callbacks = callbacks,
              epochs=EPOCHS,
              validation_data=(X_val, y_val))
    
    y_pred, rmse = model_predict(model, X_test, y_test)
    drawPlot(y_test, y_pred, rmse, output, 'KERAS', num_test)
    
    return model

def create_model_N13(features, output, EPOCHS, num_test):
    features = features.copy(deep=True)
    output = output.copy(deep=True)
    
    X_train, X_val, y_train, y_val = input_for_LSTM(features, output, 4*672, num_test)
    X_test, y_test = test_for_LSTM(features, output, num_test)
    
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1:]), activation='relu', return_sequences=True))
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=True))
    model.add(LSTM(16, activation='relu'))
    model.add(Dense(1, activation='linear'))
    print(X_train.shape[0:])   
    callbacks = [EarlyStopping(monitor='val_loss', patience = 20), 
             ModelCheckpoint(filepath = f'checkpoint/best_model N13.hdf5')]
    model.compile(
        loss='mae', # don't change it
        optimizer='adadelta',
        metrics=['accuracy'])
    model.fit(X_train,
              y_train,
              callbacks = callbacks,
              epochs=EPOCHS,
              validation_data=(X_val, y_val))
    
    y_pred, rmse = model_predict(model, X_test, y_test)
    drawPlot(y_test, y_pred, rmse, output, 'KERAS', num_test)
    
    return model

def create_model_N10(features, output, EPOCHS, num_test):
    features = features.copy(deep=True)
    output = output.copy(deep=True)
    
    X_train, X_val, y_train, y_val = input_for_LSTM(features, output, 3*672, num_test)
    X_test, y_test = test_for_LSTM(features, output, num_test)
    
    model = Sequential()
    model.add(LSTM(16, input_shape=(X_train.shape[1:]), activation='relu', return_sequences=True))
    model.add(LSTM(8, activation='relu'))
    model.add(Dense(4, activation='linear'))
    model.add(Dense(1, activation='linear'))
    callbacks = [EarlyStopping(monitor='val_loss', patience = 10), 
             ModelCheckpoint(filepath = f'checkpoint/best_model N10.hdf5')]
    model.compile(
        loss='mae', # don't change it
        optimizer='adadelta',
        metrics=['accuracy'])
    model.fit(X_train,
              y_train,
              epochs=EPOCHS,
              callbacks = callbacks,
              validation_data=(X_val, y_val))
    
    y_pred, rmse = model_predict(model, X_test, y_test)
    drawPlot(y_test, y_pred, rmse, output, 'KERAS', num_test)
    
    return model

def create_model_N8(features, output, EPOCHS, num_test):
    features = features.copy(deep=True)
    output = output.copy(deep=True)
    
    X_train, X_val, y_train, y_val = input_for_LSTM(features, output, 4*672, num_test)
    X_test, y_test = test_for_LSTM(features, output, num_test)

    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1:]), activation='relu', return_sequences=True))
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=True))
    model.add(LSTM(16, activation='relu'))
    model.add(Dense(1, activation='linear'))
    
    callbacks = [EarlyStopping(monitor='val_loss', patience = 20), 
             ModelCheckpoint(filepath = f'checkpoint/best_model N8.hdf5')]
    model.compile(
        loss='mae', # don't change it
        optimizer='adadelta',
        metrics=['accuracy'])
    model.fit(X_train,
              y_train,
              callbacks = callbacks,
              epochs=EPOCHS,
              validation_data=(X_val, y_val))
    
    y_pred, rmse = model_predict(model, X_test, y_test)
    drawPlot(y_test, y_pred, rmse, output, 'KERAS', num_test)
    
    return model

#test the loaded model (1 week test samples)
def test_loaded_model(features, output, model, num_test):
    features = features.copy(deep=True)
    output = output.copy(deep=True)
    X_test, y_test = test_for_LSTM(features, output, num_test)
    y_pred, rmse = model_predict(model, X_test, y_test)
    drawPlot(y_test, y_pred, rmse, output, 'KERAS', num_test)
    
    return model

#save the model (folder models, saving it with time stamp)
def save_model(model, make_prediction, user):
    time = datetime.datetime.now().strftime("%H_%M %d_%m_%Y")
    model.save(f'models/model - prediction for {make_prediction}  - user {user} - time {time}.hdf5')
    
def load_model_keras(name):
    return load_model(f'models/{name}.hdf5')   

## make prediction about the wanted power consumption 1 day, 1 hour, 15 mins 
def predict_wanted_power_keras(make_prediction, features, output, model, num_test, user):
    features = features.copy(deep=True)
    output = output.copy(deep=True)
    X_test, y_test = test_for_LSTM(features, output, num_test)
    a = X_test[make_prediction - 1, :, :]
    a = a.reshape(1,1,X_test.shape[0:][2])
    predict = int(model.predict(a))
    real = y_test.iloc[make_prediction-1]
    diff = abs(real - predict)
    print(f'the predicted power with Keras for {user} is {predict}, the real power is {real}, the difference is {diff}')
    return predict, real, diff

def train_and_predict_RandomForest(features, output, test_size, make_prediction):
    features = features.copy(deep=True)
    output = output.copy(deep=True)
    X_train, X_test, y_train, y_test = split_data(features, output, test_size)
    model_RF = RandomForestRegressor(n_estimators = 150)
    model_RF.fit(X_train, y_train)
    y_predict, rmse = (model_RF, X_test, y_test)
    drawPlot(y_test, y_predict, rmse, output, 'RF', test_size)

def model_features(features,user):
    if user == 'N5':
        features = features[['hour','dayOfWeek','holiday','uvIndex','minute','month']]
    if user == 'N10':
        features = features[['hour','weekend','holiday','temperature','minute','month','dewPoint','windBearing']]
    if user == 'N12':
        features = features[['hour','dayOfWeek','holiday','temperature','minute']]    
    if user == 'N13':
        features = features[['hour','dayOfWeek','holiday','temperature','minute','humidity']] 
    if user == 'N8':
        features = features[['hour','dayOfWeek','holiday','temperature']]

    return features


#   INNER functions (called in other functions)

#   split_data(features, output, num_samples)  #return X_train, X_test, y_train, y_test (called in train_and_predict_RandomForest)
#   input_for_LSTM(features, output, validation_size, test_size)    #return 3d X_train, X_test, y_train, y_test (called in create_model_NX)       
#   test_for_LSTM(features, output, test_size)  #return X_test, y_test (called in create_model_NX, test_loaded_model, predict_loaded_model)
#   evaluation_rmse(y_test, y_pred)     #return result (called in model_predict)
#   model_predict(model, X_test, y_test)    #return y_predict, rmse (called in create_model_Nx, test_loaded_models)


#   AVAILABLE FUNCTIONS
    
#   drawPlot(y_test, y_pred, score, output, name, num)
#   feature_correlation(df, output) 
#   plotcorrelation(df, output, num_of_samples)
#   add_past_values_currents(features, output, user, time_frame = 672) # return features, output
#   add_past_values_power(features, output, time_frame = 672) 
#   feature_importance_RF(features, output)
#   feature_importance_KBest(features, output)
#   create_model_N5(features, output, EPOCHS, num_test)                # return model
#   test_loaded_model(features, output, model, num_test)               # return model
#   save_model(model, make_prediction, user)
#   load_model_keras(name)                                             # return model
#   predict_wanted_power(make_prediction, features, output, model, num_test, user)
#   model_features(features,user)                                      # return features
#   train_and_predict_RandomForest(features, output, test_size, make_prediction) 



###         HERE YOU CAN ADJUST PARAMETERS          ###
  
#     Would you like to train or to test previous trained model ? If you want to train model
#    from scratch set want_train == 1 else set want_train == 0                                                         ###

want_train = 0    

test_size = 96 #96 - day,  4 - one hour, 1 - 15 min
EPOCHS = 100 
user = 'N5' # N5; N8; N10; N12; N13

num_test = test_size
if user == 'N10' or 'N12':
    validation_size = 3*672
  
if user == 'N5':
    validation_size = 4*672
     
make_prediction_for = 96 # 96 - day,  4 - hour, 1 - 15 min (used in "predict_wanted_power_keras"
#                                                           also used for adding past values)


#load_model_name = 'example here' ##look up in the folder models and choose the suitable one 
                                 ##prediction(day, hour, 15 min)  user timestamp

features = pd.read_csv(f'features/{user}_features.csv')
features = model_features(features, user)
output = pd.read_excel(f'new_data/{user}.xlsx')
output = output.interpolate()

features, output = add_past_values_currents(features, output, user, make_prediction_for)
features, output = add_past_values_power(features, output, make_prediction_for)
features, output = add_past_values_power(features, output) # default 1 week == 672

if want_train == 1:
    if user == 'N5':     
        model = create_model_N5(features, output, EPOCHS, test_size)  
    if user == 'N10':     
        model = create_model_N10(features, output, EPOCHS, test_size)  
    if user == 'N12':     
        model = create_model_N12(features, output, EPOCHS, test_size)  
    if user == 'N13':
        model = create_model_N13(features, output, EPOCHS, test_size) 
    if user == 'N8':
        model = create_model_N8(features, output, EPOCHS, test_size)
        
    save_model(model, make_prediction_for, user)  
    predict_wanted_power_keras(make_prediction_for, features, output, model, num_test, user)  

if want_train == 0: 
    
    
    model_list = os.listdir('.\\models')
    for i in model_list:
        if i.find(user) != -1 and i.find('prediction for ' + str(test_size)) != -1:
            model = load_model_keras(i[0:-5])
    
    test_loaded_model(features, output, model, num_test)
    # predict_wanted_power_keras(make_prediction_for,features,output,model,num_test,user)