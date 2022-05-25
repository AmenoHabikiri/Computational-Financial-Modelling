import matplotlib.pyplot  as plt
import numpy as np
import pandas as pd
import os
from os import listdir, getcwd, mkdir
from os.path import join, isdir, abspath
import warnings
import requests
from statsmodels.tsa.arima_model import ARIMA  
warnings.filterwarnings("ignore")
tata=pd.read_csv("TATAMOTORS.csv")
ambuja=pd.read_csv("AMBUJACEMENT.csv")
asian=pd.read_csv("ASIANPAINTS.csv")


def Returns(df):
    print(df.index)
    return [df["Close Price"][i] - df["Prev Close"][i] for i in df.index]
    # return np.array(df["Close Price"]) - np.array(df["Prev Close"])
    
tata["return"] = Returns(tata)
ambuja["return"] = Returns(ambuja)
asian["return"]= Returns(asian)
plt.plot(pd.to_datetime(tata["Date"]), tata["return"])
plt.title("for TATAMOTORS")
plt.xlabel("date")
plt.ylabel("return")
plt.show()
plt.plot(pd.to_datetime(ambuja["Date"]), ambuja["return"])
plt.title("for AMBUJA")
plt.xlabel("date")
plt.ylabel("return")
plt.show()
plt.title("for Asian")
plt.plot(pd.to_datetime(asian["Date"]), asian["return"])
plt.xlabel("date")
plt.ylabel("return")
plt.show()

data=[tata,ambuja,asian]
name = ["TATA", "AMBUJA", "ASIAN"]

from statsmodels.tsa.ar_model import AutoReg

from statsmodels.graphics.tsaplots import plot_acf
def autocorr(df):
    divide_data = int(len(df["return"])*0.8)
    train, test = df["return"][:divide_data], df["return"][divide_data:]
    values = pd.DataFrame(train)
    dataframe = pd.concat([values.shift(3), values.shift(2),
                       values.shift(1), values], axis=1)
    dataframe.columns = ['t', 't+1', 't+2', 't+3']
 
    # using corr() function to compute the correlation
    result = dataframe.corr()
    return train, test, result

#AR model for different datasets
def AR(train_data, test_data, df, i):
    ar_model = AutoReg(train_data, lags=4).fit()
    print(ar_model.summary())
    ar_pred = ar_model.predict(start=len(train_data), end=(len(df["return"])-1), dynamic=False)

    plt.plot(ar_pred)
    plt.plot(test_data, color='red')
    plt.title("AR model "+str(name[i]))
    plt.show()
# autocorrelation
for i in range(len(data)):
    # print(autocorr(i)[2])
    train_data, test_data, result = autocorr(data[i])
    #plt.acorr(autocorr(i)[0])
    plot_acf(train_data)
    plt.title("Autocorrelation of "+str(name[i]))
    plt.show()
    AR(train_data, test_data, data[i], i)
    

#ARIMA model
seas_diff= pd.Series(tata["return"])
model = ARIMA(np.asarray(seas_diff), order=(0, 1, 0))
model_fit = model.fit(disp=0)
print("TATA \n",model_fit.summary())

# Build Model
# model = ARIMA(, order=(0,1,0))
model = ARIMA(autocorr(tata)[0], order=(0, 1, 0))
fitted = model.fit(disp=-1)

# Forecast
forecast, se, conf = fitted.forecast(51, alpha=0.05)  # 95% conf
# Make as pandas series
forecast_series = pd.Series(forecast, index=autocorr(tata)[1].index)
lower_series = pd.Series(conf[:, 0], index=autocorr(tata)[1].index)
upper_series = pd.Series(conf[:, 1], index=autocorr(tata)[1].index)

# Plot
plt.plot(autocorr(tata)[0], label='training')
plt.plot(autocorr(tata)[1], label='actual')
plt.plot(forecast_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series,
                 color='red', alpha=.15)
plt.title('Forecast vs Actuals for seasonal differenced for tata')
plt.ylabel("Value")
plt.legend(loc='upper left', fontsize=5)
plt.show()

#ARIMA model
seas_diff= pd.Series(ambuja["return"])
model = ARIMA(np.asarray(seas_diff), order=(0, 1, 0))
model_fit = model.fit(disp=0)
print("TATA \n",model_fit.summary())

# Build Model
# model = ARIMA(, order=(0,1,0))
model = ARIMA(autocorr(ambuja)[0], order=(0, 1, 0))
fitted = model.fit(disp=-1)

# Forecast
forecast, se, conf = fitted.forecast(50, alpha=0.05)  # 95% conf
# Make as pandas series
forecast_series = pd.Series(forecast, index=autocorr(ambuja)[1].index)
lower_series = pd.Series(conf[:, 0], index=autocorr(ambuja)[1].index)
upper_series = pd.Series(conf[:, 1], index=autocorr(ambuja)[1].index)

# Plot
plt.plot(autocorr(ambuja)[0], label='training')
plt.plot(autocorr(ambuja)[1], label='actual')
plt.plot(forecast_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series,
                 color='red', alpha=.15)
plt.title('Forecast vs Actuals for seasonal differenced for Ambuja')
plt.ylabel("Value")
plt.legend(loc='upper left', fontsize=5)
plt.show()

#ARIMA model
seas_diff= pd.Series(asian["return"])
model = ARIMA(np.asarray(seas_diff), order=(0, 1, 0))
model_fit = model.fit(disp=0)
print("TATA \n",model_fit.summary())

# Build Model
# model = ARIMA(, order=(0,1,0))
model = ARIMA(autocorr(asian)[0], order=(0, 1, 0))
fitted = model.fit(disp=-1)

# Forecast
forecast, se, conf = fitted.forecast(50, alpha=0.05)  # 95% conf
# Make as pandas series
forecast_series = pd.Series(forecast, index=autocorr(asian)[1].index)
lower_series = pd.Series(conf[:, 0], index=autocorr(asian)[1].index)
upper_series = pd.Series(conf[:, 1], index=autocorr(asian)[1].index)

# Plot
plt.plot(autocorr(asian)[0], label='training')
plt.plot(autocorr(asian)[1], label='actual')
plt.plot(forecast_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series,
                 color='red', alpha=.15)
plt.title('Forecast vs Actuals for seasonal differenced for Ambuja')
plt.ylabel("Value")
plt.legend(loc='upper left', fontsize=5)
plt.show()