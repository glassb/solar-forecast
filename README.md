# Solar Energy Production Forecaster

### Overview
This Python program uses a linear regression model to produce a short-term energy production forecast for a residential rooftop PV installation. The model uses historical solar energy output information and weather data to produce an estimate for future production values.

### Model
Multi-variable linear regression model

**Inputs:**
- Current 15-min Solar Production Value (Wh)
- Solar Irradiance (W/m2)
- Month
- Time

**Outputs:**
- Next 15-min Solar Production Value (Wh)

Note: for forecasting, the predicted solar production value is plugged into the model as the current solar production value. 

### Data
**Rooftop PV data:** I acquired Rooftop PV data at 15-minute intervals from the last 2 years. I trained the linear regression model using data from Dec 2022 until November 2024. I tested the model using data from 2 weeks in December 2024.

**Weather data:** I acquired solar irradiance data at 15-min intervals from [The Weather Scraper](https://github.com/Karlheinzniebuhr/the-weather-scraper) (created by [Karlheinzniebuhr](https://github.com/Karlheinzniebuhr)). 

### Notes
This was a personal project for an engineering workshop at Boston University, January 2025. 
