'''

Objective: predict energy output at the next timestamp based on current variables (weather, hour of the day, month, etc.).

Output: prediction [Joules]

Input: Vector X [Month, Time of Day, Cloud Cover, Current Energy Output]

Model: Vector LSSE (multivariate regression)

'''

# ------------------------------- FRONT MATTER -------------------------------

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import warnings

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

'''
df_solar = pd.read_csv([REDACTED PATH],sep=',')

# ------------------------------- DATA PRE PROCESSING -------------------------------

# I used this script/package to scrape wunderground for weather data in [REDACTED]
#https://github.com/Karlheinzniebuhr/the-weather-scraper?tab=readme-ov-file

# ------------------------------- DATA PRE PROCESSING -------------------------------

#CLEANED PV ENERGY OUTPUT DATASET
df_solar_cleaned = pd.DataFrame([])
df_solar_cleaned["Date"] = df_solar["Date/Time"].str.split(' ').str[0]
df_solar_cleaned["Date"] = pd.to_datetime(df_solar_cleaned["Date"])
df_solar_cleaned['Date'] = df_solar_cleaned['Date'].dt.strftime('%Y/%m/%d')
df_solar_cleaned['Date'] = df_solar_cleaned['Date'].astype('str')
df_solar_cleaned["Time"] = df_solar["Date/Time"].str.split(' ').str[1]
df_solar_cleaned["Month"] = df_solar["Date/Time"].str.split('/').str[0]
df_solar_cleaned['EnergyProd-Wh'] = df_solar["Energy Produced (Wh)"]

#WEATHER DATASET ACQUIRED USING THE-WEATHER-SCRAPER package (from Karlheinzniebuhr)
df_weather = pd.read_csv([REDACTED PATH],sep=',')

df_weather_cleaned = pd.DataFrame([])
df_weather_cleaned["Date"] = df_weather["Date"].astype(str)
df_weather_cleaned["Time"] = df_weather["Time"]
df_weather_cleaned["Hold"] = df_weather["Time"].str.split(':').str[1]
df_weather_cleaned["Hour"] = df_weather["Time"].str.split(':').str[0].astype(int)
df_weather_cleaned["AM/PM"] = df_weather_cleaned["Hold"].str.split(' ').str[1]
df_weather_cleaned["Minute"] = df_weather_cleaned["Hold"].str.split(' ').str[0].astype(int)
df_weather_cleaned['UpdatedHour'] = df_weather_cleaned["Hour"]
#df_weather_cleaned["UpdatedHour"].loc[df_weather_cleaned['AM/PM'] == 'PM'] = df_weather_cleaned["Hour"] + 12
df_weather_cleaned["UpdatedHour"].loc[df_weather_cleaned['AM/PM'] == 'PM'] = df_weather_cleaned["Hour"] + 12
df_weather_cleaned["UpdatedHour"].loc[(df_weather_cleaned['UpdatedHour'] == 24) & (df_weather_cleaned['AM/PM'] == 'PM')] = 12
df_weather_cleaned["UpdatedHour"].loc[(df_weather_cleaned['UpdatedHour'] == 12) & (df_weather_cleaned['AM/PM'] == 'AM')] = 0
df_weather_cleaned["UpdatedMinute"] = (df_weather_cleaned["Minute"]+1).astype(str).str.pad(width=2, side='left', fillchar='0')
df_weather_cleaned["UpdatedHour"].loc[df_weather_cleaned['UpdatedMinute'] == '60'] = df_weather_cleaned["UpdatedHour"] + 1
df_weather_cleaned["UpdatedMinute"].loc[df_weather_cleaned['UpdatedMinute'] == '60'] = '00'
df_weather_cleaned["UpdatedHour"].loc[(df_weather_cleaned['UpdatedHour'] == 24)] = 0
df_weather_cleaned["UpdatedHour"] = df_weather_cleaned["UpdatedHour"].astype(str).str.pad(width=2, side='left', fillchar='0')
df_weather_cleaned['UpdatedTS'] = df_weather_cleaned["UpdatedHour"]+':'+df_weather_cleaned["UpdatedMinute"]
df_weather_cleaned["OrigTime"] = df_weather["Time"]
df_weather_cleaned["Time"] = df_weather_cleaned["UpdatedTS"]
df_weather_cleaned['Solar_w/m2'] = df_weather['Solar_w/m2']
df_weather_cleaned['Temperature_F'] = df_weather['Temperature_F']
df_weather_cleaned['UV'] = df_weather['UV']


#MERGING PV AND WEATHER DATASETS
df_merged = pd.merge(df_solar_cleaned, df_weather_cleaned, on=['Date','Time'])
df_merged = df_merged.loc[:, ['Date','Time','Month','EnergyProd-Wh','Solar_w/m2','Temperature_F','UV']]
df_merged["Hour"] = df_merged["Time"].str.split(':').str[0].astype(int)
#df_merged["EnergyProd-Wh"] = df_merged["EnergyProd-Wh"]
df_merged["Timestamp"] = df_merged['Date'].astype(str)+'-'+df_merged['Time']
pd.to_datetime(df_merged['Timestamp'], format="%Y/%m/%d-%H:%M").sort_values()

# ADD PREDICITVE ENERGY VALUE
df_merged['Next-Energy-Value(Wh)'] = ""
df_merged['Energy_Difference'] = ""
for i in range(0,len(df_merged)-1):
	#print(df_site_66.iloc[i,4]);
	prev_value = df_merged.iloc[i-10,3]
	fut_value = df_merged.iloc[i+1,3]
	#print(fut_value)
	slope = (df_merged.iloc[i,3]-prev_value)
	df_merged.iloc[i,9] = fut_value
	df_merged.loc[i,'Energy_Difference'] = slope
	#print(slope)

# ADD Previous Period's Energy value
df_merged['LastPeriodEnergyVal'] = ""
#print(df_merged)
for i in range(0,len(df_merged)-1):
	#print(df_site_66.iloc[i,4]);
	prev_value = df_merged.iloc[i-96,3]
	#fut_value = df_merged.iloc[i+1,3]
	#print(fut_value)
	#slope = (fut_value-prev_value)/2
	df_merged.iloc[i,10] = prev_value
	#print(slope)

# data cleaning commands
df_merged['Next-Energy-Value(Wh)'] = pd.to_numeric(df_merged['Next-Energy-Value(Wh)'])
pd.to_numeric(df_merged['LastPeriodEnergyVal'])
#print(df_merged['Next-Energy-Value(Wh)'].head(10))
df_merged = df_merged.dropna()
#print(df_merged.head(100))
df_merged_grouped = df_merged.groupby('Date')
df_merged_grouped = df_merged_grouped['EnergyProd-Wh'].sum()
#print(df_merged_grouped.describe())


df_merged['Cloud_Index'] = ''
df_merged['EnergyProd-Wh-Daily-Sum'] = ''
for i in range(0,len(df_merged)-1):
	df_temp = df_merged.iloc[i]
	df_temp_date = df_temp['Date']
	#print(df_temp_date)

	df_temp_solar_sum = df_merged.loc[df_merged['Date'] == df_temp_date,'EnergyProd-Wh'].sum()
	df_merged.loc[i,'EnergyProd-Wh-Daily-Sum'] = df_temp_solar_sum
	#print(df_temp_date_sum)


	if df_temp_solar_sum >= 47500:
		df_merged.loc[i,'Cloud_Index'] = 'Sunny'
	elif df_temp_solar_sum >= 15000:
		df_merged.loc[i,'Cloud_Index'] = 'Cloudy'
	elif df_temp_solar_sum > 0:
		df_merged.loc[i,'Cloud_Index'] = 'VeryCloudy'

	if df_merged.loc[i,'EnergyProd-Wh'] == 0 :
		df_merged.loc[i,'Cloud_Index'] = 'Dark'



#SPLIT TESTING AND TRAINING DATA
df_merged_testing = df_merged.tail(5000)
df_merged = df_merged[:-5000]
#print(df_merged_testing)


df_merged.to_csv([REDACTED PATH], index=True)
df_merged_testing.to_csv([REDACTED PATH], index=True)
'''


df_merged = pd.read_csv([REDACTED PATH],sep=',')
df_merged_testing = pd.read_csv([REDACTED PATH],sep=',')
#print(df_merged.head())



#CONFIRMING THAT TRAINING AND TEST DATASETS DO NOT SHARE DATA
test = pd.merge(df_merged, df_merged_testing, on=['Date','Time'])
print('OVERLAP TEST')
print(test)

print(df_merged[['Date','Time']].tail(10))
print(df_merged_testing[['Date','Time']].head(10))


'''
sns.displot(data=df_merged,x='EnergyProd-Wh-Daily-Sum',kind="kde")
plt.show()
'''


# ------------------------------- MODEL BUILDING -------------------------------

# NOTE: the code below was created by Professor Nazer for BU's EK381. I received permission to use this code and alter it for this project.

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# SUNNY MODEL
sunny_df = df_merged[df_merged['Cloud_Index'] == 'Sunny']

# split data into sub-test and sub-train datasets 
features_train, features_test, target_train, target_test = (
    train_test_split(sunny_df[["Month","Hour",'Solar_w/m2','EnergyProd-Wh','Energy_Difference','Temperature_F','UV']], sunny_df["Next-Energy-Value(Wh)"], test_size=0.2))

#"Month","Hour",'Solar_w/m2','EnergyProd-Wh','Temperature_F','UV'
#"Month","Hour",'Solar_w/m2',
#"Month","Hour",'Solar_w/m2','Temperature_F','UV','EnergyProd-Wh'

# model training (MLR)
sunny_model = LinearRegression()
sunny_model.fit(features_train, target_train)
print('\n\n')
print('SUNNY MODEL')
print("Model Coefficients: ", sunny_model.coef_)
print("Model Intercept: ",sunny_model.intercept_)

# modelling prediction
target_predict = sunny_model.predict(features_test)

# model metrics
MSE = mean_squared_error(target_test, target_predict)
R2 = r2_score(target_test, target_predict)
SE = np.sqrt(((target_test-target_predict) ** 2).sum()/(len(df_merged)-1))

# Output results
print('Standard Error: ',SE)
print("Mean Squared Error:", MSE)
print("R-squared:", R2)




# CLOUDY MODEL
cloudy_df = df_merged[df_merged['Cloud_Index'] == 'Cloudy']

# split data into sub-test and sub-train
features_train, features_test, target_train, target_test = (
    train_test_split(cloudy_df[["Month","Hour",'Solar_w/m2','EnergyProd-Wh','Energy_Difference','Temperature_F','UV']], cloudy_df["Next-Energy-Value(Wh)"], test_size=0.2))

#"Month","Hour",'Solar_w/m2','EnergyProd-Wh','Temperature_F','UV'
#"Month","Hour",'Solar_w/m2',
#"Month","Hour",'Solar_w/m2','Temperature_F','UV','EnergyProd-Wh'

# model training (MLR)
cloudy_model = LinearRegression()
cloudy_model.fit(features_train, target_train)
print('\n\n')
print('CLOUDY MODEL')
print("Model Coefficients: ", cloudy_model.coef_)
print("Model Intercept: ",cloudy_model.intercept_)

# modelling prediction
target_predict = cloudy_model.predict(features_test)

# model metrics
MSE = mean_squared_error(target_test, target_predict)
R2 = r2_score(target_test, target_predict)
SE = np.sqrt(((target_test-target_predict) ** 2).sum()/(len(df_merged)-1))

# Output results
print('Standard Error: ',SE)
print("Mean Squared Error:", MSE)
print("R-squared:", R2)


# VERY CLOUDY MODEL
verycloudy_df = df_merged[df_merged['Cloud_Index'] == 'VeryCloudy']

#print(verycloudy_df)

# split data into sub-test and sub-train
features_train, features_test, target_train, target_test = (
    train_test_split(verycloudy_df[["Month","Hour",'Solar_w/m2','EnergyProd-Wh','Energy_Difference','Temperature_F','UV']], verycloudy_df["Next-Energy-Value(Wh)"], test_size=0.2))

#"Month","Hour",'Solar_w/m2','EnergyProd-Wh','Temperature_F','UV'
#"Month","Hour",'Solar_w/m2',
#"Month","Hour",'Solar_w/m2','Temperature_F','UV','EnergyProd-Wh'

# model training (MLR)
verycloudy_model = LinearRegression()
verycloudy_model.fit(features_train, target_train)
print('\n\n')
print('VERY CLOUDY MODEL')
print("Model Coefficients: ", verycloudy_model.coef_)
print("Model Intercept: ",verycloudy_model.intercept_)

# modelling prediction
target_predict = verycloudy_model.predict(features_test)

# model metrics
MSE = mean_squared_error(target_test, target_predict)
R2 = r2_score(target_test, target_predict)
SE = np.sqrt(((target_test-target_predict) ** 2).sum()/(len(df_merged)-1))

# Output results
print('Standard Error: ',SE)
print("Mean Squared Error:", MSE)
print("R-squared:", R2)


# ------------- where code ends from Prof. Nazer






# ------------------------------- MODEL TESTING -------------------------------


df_merged = df_merged_testing[:-1]


energy_predict = 0
df_prediction = []

#test one week's worth of test data (672 rows)
start = 96
end = 768
for i in range(start,end):
	df_temp = df_merged.iloc[i]
	df_temp = df_temp[["Month","Hour",'Solar_w/m2','EnergyProd-Wh','Energy_Difference','Temperature_F','UV']]
	actual_value = df_temp[['EnergyProd-Wh']]
	prev_value = df_merged.loc[i-96,'EnergyProd-Wh']
	df_temp['EnergyProd-Wh'] = energy_predict

	#"Month","Hour",'Solar_w/m2','Temperature_F','UV','EnergyProd-Wh'
	if df_merged.loc[i,'Cloud_Index'] == 'Sunny':
		prediction = sunny_model.predict(df_temp.to_numpy().reshape(1,-1))
	elif df_merged.loc[i,'Cloud_Index'] == 'Cloudy':
		prediction = cloudy_model.predict(df_temp.to_numpy().reshape(1,-1))
	elif df_merged.loc[i,'Cloud_Index'] == 'VeryCloudy':
		prediction = verycloudy_model.predict(df_temp.to_numpy().reshape(1,-1))
	elif df_merged.loc[i,'Cloud_Index'] == 'Dark':
		prediction = np.array([0])

	energy_predict = prediction

	#zeroing the prediction
	if prev_value <= 0 or energy_predict[0] <= 0:
		energy_predict[0] = 0
	else:
		energy_predict = prediction


	df_temp['Prediction'] = energy_predict[0]
	df_temp['Residual'] = df_merged.loc[i,'EnergyProd-Wh'] - energy_predict[0]
	df_temp['Timestamp'] = df_merged.loc[i, 'Timestamp']
	
	df_prediction.append(df_temp)
	

df_prediction = pd.DataFrame(df_prediction)
df_prediction = df_prediction[['Timestamp','Prediction']]
df_prediction_merged = pd.merge(df_merged, df_prediction, on=['Timestamp'])
df_prediction_merged['Residual'] = df_prediction_merged['EnergyProd-Wh'] - df_prediction_merged['Prediction']



# ------------------------------- MODEL TESTING OUTPUTS (PLOT, METRICS, etc.) -------------------------------

# Test data plot
sns.lineplot(x=df_prediction_merged['Timestamp'], y=df_prediction_merged['EnergyProd-Wh'], label='Testing Data')
sns.lineplot(x=df_prediction_merged['Timestamp'], y=df_prediction_merged['Prediction'], label='Model Forecast', linewidth=2)
sns.set_theme(style="darkgrid")
plt.xlabel('Timestamp',fontweight='bold')
plt.ylabel('Energy Production (Wh)',fontweight='bold')
plt.title('7-day Forecast on unseen test data from April 30 - May 7, 2024',fontweight='bold')
plt.fill_between(df_prediction_merged['Timestamp'], df_prediction_merged['EnergyProd-Wh'], 0, alpha=0.3)
plt.xticks(df_prediction_merged['Timestamp'], df_prediction_merged['Timestamp'], rotation=15)
plt.locator_params(axis="x", nbins=7) #this line of code was created by AndysPythonStuff here: https://stackoverflow.com/questions/73793171/reducing-the-number-of-labels-on-x-axis-of-plot
plt.legend()
plt.show()

# Testing statistics
print('\n\n')
print('PREDICTION STATS')
print('Total Residual: ', df_prediction_merged['Residual'].abs().sum())
print('Error Percentage: ', 1-(df_prediction_merged['Prediction'].abs().sum()/df_prediction_merged['EnergyProd-Wh'].sum()))



			


	






