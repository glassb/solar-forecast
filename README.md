# Solar Energy Production Forecaster

This Python program generates a short-term energy production forecast for a residential rooftop PV system. The program uses a set of linear regression models to estimate energy production 15 minutes into the future based on current metrics. Predictions are recursivelly fed back to the model to generate a longer term forecast.


## Results
Tested on data from a 7-day period in early May 2025, the model forecasted total energy production with 97-99% accuracy (depending on model run).


<img width="1060" alt="Screenshot 2025-01-08 at 15 50 06" src="https://github.com/user-attachments/assets/5ab7406f-fd7d-470a-aaaf-74f40518df6a" />
<img width="1132" alt="Screenshot 2025-01-08 at 15 50 39" src="https://github.com/user-attachments/assets/75796129-7ef6-479b-b0f9-9afd81c3b008" />


## Model
Multiple Linear Regression

**Inputs:**
- Current 15-min Solar Production Value (Wh)
- Solar Irradiance (W/m2)
- Month
- Hour of Day
- Temperature (Fahreinheit)
- UV (Unit unknown)
- Energy Difference (difference between current energy production value and the energy value from 10 time-steps before)

**Outputs:**
- Next 15-min Solar Production Value (Wh)

There are three MLR models for sunny, cloudy, and very cloudy days (bins were arbitrarily determined based on solar irradiance data distribution). 

Note: In order to forecast future outputs, I feed solar production predictions back into the model. For example, in order to predict the solar production value at 2 timestamps ahead, I first predict the solar production at 1 timestamp ahead, then feed that prediction as the current production value for the prediction at 2 timestamps ahead.


## Data
**Rooftop PV data:** I acquired Rooftop PV data at 15-minute intervals from the last 2 years. The linear regression model were trained on data from Dec 2022 until April 2024. I tested the model using data from 1 week in May 2024.

**Weather data:** I acquired solar irradiance data at 15-min intervals from [The Weather Scraper](https://github.com/Karlheinzniebuhr/the-weather-scraper) (created by [Karlheinzniebuhr](https://github.com/Karlheinzniebuhr)). 

## Limitations/Areas for Further Development
This model requires high quality solar irradiance data at each time step, which cannot realistically be provided for forecasting. New models should be configured to generate forecasts based on data that be feasibly acquired. This could be achieved by using something less granular such as AM/PM weather type predictions (e.g. sunny, cloudy, partly cloudy, etc.).

## Notes
This was a personal project for an engineering workshop at Boston University, January 2025. 
