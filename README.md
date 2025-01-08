# Solar Energy Production Forecaster

This Python program uses a linear regression model to produce a short-term energy production forecast for a residential rooftop PV installation. The model uses historical solar energy output information and weather data to produce an estimate for future production values.


## Tentative Results (Unverified)
Tested on a 7-day period in early May 2025, the model forecasted energy production with 97-99% accuracy (depending on model run).

<img width="881" alt="Screenshot 2025-01-08 at 12 57 01" src="https://github.com/user-attachments/assets/055c718c-4d67-4ae8-93c7-286576e50644" />



## Model
Multiple Linear Regression

**Inputs:**
- Current 15-min Solar Production Value (Wh)
- Solar Irradiance (W/m2)
- Month
- Time of Day
- Temperature

**Outputs:**
- Next 15-min Solar Production Value (Wh)

Note: In order to forecast future outputs, I feed solar production predictions back into the model. For example, in order to predict the solar production value at 2 timestamps ahead, I first predict the solar production at 1 timestamp ahead, then feed that prediction as the current production value for the prediction at 2 timestamps ahead.


## Data
**Rooftop PV data:** I acquired Rooftop PV data at 15-minute intervals from the last 2 years. I trained the linear regression model using data from Dec 2022 until April 2024. I tested the model using data from 2 weeks in December 2024.

**Weather data:** I acquired solar irradiance data at 15-min intervals from [The Weather Scraper](https://github.com/Karlheinzniebuhr/the-weather-scraper) (created by [Karlheinzniebuhr](https://github.com/Karlheinzniebuhr)). 

## Notes
This was a personal project for an engineering workshop at Boston University, January 2025. 
