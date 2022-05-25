import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import math
from pandas import datetime
import numpy as np
from statsmodels.tsa.arima_model import ARIMA  
from scipy.stats import pearsonr
import warnings
import requests

warnings.filterwarnings("ignore")


series = pd.read_csv('electricity.csv')
series = series.astype({"YYYYMM": str})
series["YYYYMM"] = pd.to_datetime([str(i[-2:]+"-"+str(i[:4])) for i in series["YYYYMM"]])
print("Time series Plot\n")
plt.title("Time series data of original dataset")
plt.plot(series["YYYYMM"],series["Value"])
plt.ylabel("Values")
plt.xlabel("Month")
plt.show()

##################################################

diff=[]
for i in range(1,len((series["Value"]))):
    diff.append((series["Value"])[i]-(series["Value"])[i-1])
plt.title("Differenced values plot(Lag=1")
plt.ylabel("difference Values")

###################################################
plt.plot(diff)
plt.show()
# Creating Autocorrelation plot
x = pd.plotting.autocorrelation_plot(diff)
plt.title("Autocorrelation of differenced series")
# plotting the Curve
x.plot()
 
# Display
plt.show()

#####################################################
#for seasonal difference
seas_diff=[]
for i in range(1,len((series["Value"]))):
    seas_diff.append((series["Value"])[i]-(series["Value"])[abs(i-12)])
plt.title("seasonal differenced values plot")
plt.ylabel("Seasonal differenced Values")
plt.plot(seas_diff)
plt.show()

# Creating Autocorrelation plot for original data
x = pd.plotting.autocorrelation_plot(seas_diff)
plt.title("Autocorrelation of seasonal differenced set")
# plotting the Curve
x.plot()
 
# Display
plt.show()

print("iv\n")
#differenced and seasonal differenced series of data plot
plt.title("seasonal differenced and differenced values plot")
plt.plot(diff)
plt.plot(seas_diff)
plt.show()

######################################
#ARMA model
# 0,1,0 ARIMA Model
# seas_diff = [str(i) for i in seas_diff]
diff= pd.Series(diff)
model = ARIMA(np.asarray(diff), order=(0, 1, 0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
train = diff.dropna()[:350]  # first 350 rows in the training set
test =diff.dropna()[350:]  # remaining rows in the test set



#ARMA model
# 0,1,0 ARIMA Model
# seas_diff = [str(i) for i in seas_diff]
seas_diff= pd.Series(seas_diff)
model = ARIMA(np.asarray(seas_diff), order=(0, 1, 0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
train = seas_diff.dropna()[:350]  # first 350 rows in the training set
test =seas_diff.dropna()[350:]  # remaining rows in the test set

# Build Model
# model = ARIMA(, order=(0,1,0))
model = ARIMA(seas_diff[:350], order=(0, 1, 0))
fitted = model.fit(disp=-1)

# Forecast
forecast, se, conf = fitted.forecast(33, alpha=0.05)  # 95% conf
# Make as pandas series
forecast_series = pd.Series(forecast, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(forecast_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series,
                 color='red', alpha=.15)
plt.title('Forecast vs Actuals for seasonal differenced')
plt.ylabel("Value")
plt.legend(loc='upper left', fontsize=5)
plt.show()
#SARIMAX model
from statsmodels.tsa.statespace.sarimax import SARIMAX
  
model = SARIMAX(train, 
                order = (0, 1, 1), 
                seasonal_order =(2, 1, 1, 12))
  
result = model.fit()
result.summary()
predictions = result.predict(len(train), len(train) + len(test) - 1,
                             typ = 'levels').rename("Predictions")
  
# plot predictions and actual values
predictions.plot(legend = True)
test.plot(legend = True)
plt.title("SARMIX Model")
plt.show()
