import yfinance as yf
import pandas as pd
import datetime
from prophet import Prophet
 # pystan library



df = yf.download("QQQ", start="2002-01-01", end="2022-12-01")

df.reset_index(inplace=True)

#only keep columns 'Date' and 'Close'
df = df[['Date', 'Close']]

# rename columns
df.columns = ['ds','y']

# train model
m = Prophet(interval_width=0.95, daily_seasonality=True)
model = m.fit(df)

# forecast
future = m.make_future_dataframe(periods=365, freq='D') #where the frequency is one day
forecast = m.predict(future)
forecast.head()
forecast.tail()

plot1 = m.plot(forecast)

plot2 = m.plot_components(forecast)