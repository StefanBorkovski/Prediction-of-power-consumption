## FUNCTIONS EXPLANATION:
---
### “Input_for_LSTM”  
> This function takes two-dimensional matrices (features and output data) and reshapes them into three-dimensional matrices as needed for the LSTM layers. Also, the function normalizes the data in the range from zero to one and splits it into training data and validation data.
### “Test_for_LSTM”  
> This function is similar to “Input_for_LSTM”, but the output generated here is used for testing purposes.
### “Evaluation_rmse “ 
> This function is used for calculating the mean square root error between predicted values and test values.
### “Split_data” 
> When a random tree regressor is used for prediction, this function is used for splitting the data into training and testing data. 
### “Model_predict” 
> This function is used for loading previously trained models and evaluating the mean squared error.
### “Draw_plot”
> This function prints out the real and the predicted values on the same graph, and the error between them.
### “Add_past_values_currents” 
> This function is used for adding the past current values to the training data. Depending on the predicting period the function adds the past values from 15 minutes, 1 hour, or 1 day.
### “Add_past_values_power” 
> This function is similar to “Add_past_values_currents” and is used for adding the past power values to the training data. 
### “Features_importance_RF&feature_importance_KBest”
> This function is used for evaluating the feature importance. It looks for correlations between features and outputs. If variations in some feature values cause variation in output then the selected feature is classified as more important than others.
### “Create_model_N(number)” 
> This function is used to create the structure of the LSTM network and to train the model.
### “Test_loaded_model” 
> This function is used to test the previously created model. The function predicts the outputs for the chosen prediction time frame and prints out the real and the predicted power. 
### “Save_model” 
> This function is used to save the trained model in a folder named “models”. The name of the model consists of a user name, prediction time frame, and time when the model is created. 
### “Predict_wanted_power_keras”  
> This function is used to predict power consumption for the next 15 minutes, 1 hour, or a day. It prints out the real and the predicted value of power consumption and the error between them.
### “Train_and_predict_RandomForest"
> This function is used to predict outputs with “Random Forest Regressor”.
### “Model_features” 
> This function prints out the features used for training the network for every selected user. Every user is using a different set of features for training.
