# Prediction of power consumption: Project explanation:

The main goal of the project was to create a model from time-series data that can be used for the prediction of future electricity consumption. There are five electrical energy consumers given as “N(number).json”. Also, there is a “description.csv” file where an explanation for every user parameter is given as well as “weather_data.csv” and “static Features.csv” files that contain additional information that can be used for the model creation. The project task was to analyze the data of every user, create an appropriate model, and to make a prediction for the current (electricity) consumption. When the prediction is made, three cases should be considered:
- to predict for the next 15 minutes 
- to predict for the next hour (if the value of the consumption for 3 pm is known, the prediction should be made till 4 pm)
- to predict for the next day (if the value of the consumption for 3 pm today is known, the prediction should be made till 3 pm the next day)

## FILES EXPLANATION:
---
There are three codes: “allusers_feature_extraction.py”; “FinalCode.py”; “generateNewData.py”.
### “preprocessing_data.py”
> This code is used for data type conversion from JSON to XLSX and also for initially feature selection. 
### “features_selection.py” 
> This code is used for extracting features. In this process, only features that are highly correlated to the output data (whose variations cause changes in the output data) are chosen. 
### “functions_code.py”
> This code consists of functions that are described in "functions_explanation.md" file.

** **Unfortunately only the data from the "N5" user is uploaded because it was received from project partners.**

** **To test the algorithm the one can use “functions_code.py”. Prediction parameters such as user number, number of epochs, and prediction frame length can be adjusted in the code section from the 413-th line in “functions_code.py”.**
