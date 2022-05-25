import matplotlib.pyplot as plt
import csv
import pandas as pd
from datetime import datetime
import numpy as np
from numpy import log,sqrt
from statsmodels.tsa.stattools import adfuller
import math
from pandas import read_csv
from scipy.stats import boxcox

###########################################################################################

file=open('Passenger cars.csv')
csvreader=csv.reader(file)
li=[]
for reader in csvreader:
    # print(reader)
    li.append(reader)
li1=[]
li2=[]
for i in range(2,len(li[2])):
    li1.append(li[2][i])
    li2.append(int(li[3][i]))
my_year=[int(i[:4]) for i in li1]
my_month=[int(i[-2:]) for i in li1]
d=[]
for i in range(0,len(my_year)):
    d.append(datetime(year=my_year[i], month=my_month[i],day=1).date())
df = pd.DataFrame({'date':d,'Passenger Cars in use': li2})
#creating csv file of timeseries data
fields=['Date','Number of Cars in use']
rows=[]
for i in range(0,len(li2)):
    k=[]
    k.append(d[i])
    k.append(li2[i])
    rows.append(k)
with open('GFG', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(rows)
series = read_csv('GFG', header=0, index_col=0, squeeze=True)
X = series.values
result = adfuller(X)
print("Fuller test hypothesis\n")
print("p-value > 0.05: Fail to reject the null hypothesis (H0), the data has a unit root and is non-stationary")
print("p-value <= 0.05: Reject the null hypothesis (H0), the data does not have a unit root and is stationary\n")
print("Fuller test on original sataset:")
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

plt.plot(df["date"], df["Passenger Cars in use"])
plt.xlabel("Date")
plt.ylabel("Number of Passenger Cars in use")
plt.title("Time Series data of original dataset")
plt.show()#time series line plot
pd.plotting.autocorrelation_plot(df["Passenger Cars in use"])
plt.title("Autocorrelation Plot for original Dataset")
plt.show()

##########################################################

diff=[]
plt.title("Differenced Time Series(k=1)")
plt.xlabel("Indices")
plt.ylabel("Difference")
for i in range(1,len(li2)):
    diff.append(li2[i]-li2[i-1])
plt.plot(diff)
plt.show()
plt.title("Autocorrelation Plot Differenced Time Series(k=1)")
pd.plotting.autocorrelation_plot(diff)
plt.show()

###################################################################

diff2=[]
mean=[]
li3=[]
for i in range(0,len(li2),12):
    m=0
    for j in range(i,i+12):
        m+=li2[i]
    mean.append(m/12)
k=0
for i in range(1,len(li2)):
    if (i%12==0):
        k+=1
    li3.append(li2[i]-mean[k])
for i in range(1,len(li3)):
    diff2.append(li3[i]-li3[i-1])
plt.title("Rolling Mean Differenced Time Series(k=12)")
plt.xlabel("Indices of months in 12 month interval")
plt.ylabel("Difference")
plt.plot(diff2)
plt.show()
pd.plotting.autocorrelation_plot(diff2)
plt.title("Autocorrelation Rolling Mean Differenced Time Series(k=12)")
plt.show()

#############################################################

diff3=[]
for i in range(1,len(li2)):
    diff3.append(math.log(li2[i])-math.log(li2[i-1]))
plt.title("Log transformed Differenced Time Series")
plt.xlabel("Indices")
plt.ylabel("Difference")
plt.plot(diff3)
plt.show()
plt.title("Autocorrelation Plot of Log transformed Differenced Time Series")
pd.plotting.autocorrelation_plot(diff3)
plt.show()

############################################################################


#Applying Moving Window Function on three different frames 3 months, 6 months, 12 months
df['MA06']=df['Passenger Cars in use'].rolling(6).mean()
df['MA12']=df['Passenger Cars in use'].rolling(12).mean()
df['MA30']=df['Passenger Cars in use'].rolling(30).mean()
plt.style.use('default')  
df[['Passenger Cars in use','MA06','MA12','MA30']].plot(label='rolling mean on passenger cars in use', figsize=(16, 8))
#plt.show()
#value shows best results for MA12, seasonality is removed from the data set
#now using log and pow transformation on the data
df['LOG'] = log(df['Passenger Cars in use'])
df['SquareRoot'] = sqrt(df['Passenger Cars in use'])
#df['POW']=boxcox(df['Passenger Cars in use'],lmbda=0.0)
df['LOG'] = log(df['Passenger Cars in use'])
df['SR'] = sqrt(df['Passenger Cars in use'])
#df[['LOG','SquareRoot',[]]].plot(label='Plot of LOG and X^0.25 function', figsize=(16, 8))
plt.title('Plot of Moving Average with original dataset : MA12 : Window size 12. MA06 : Window size 6. MA30 : Window size 30.')
plt.show()
df['DiffMA12'] = df['Passenger Cars in use']-df['MA12']
df['LOGMA12'] = df['LOG'].rolling(12).mean()
df['DiffLOGMA12']=df['LOG']-df['LOGMA12']
df['SRMA12'] = df['SR'].rolling(12).mean()
df['DiffSRMA12']=df['SR']-df['SRMA12']
df[['LOG','LOGMA12']].plot(label='rolling mean on passenger cars in use', figsize=(16, 8))
plt.title("Plot of Log transformed Differenced Time Series vs Moving Average(k=12 months) Log transformed Differenced Time series data")
plt.show()
df[['SRMA12','DiffSRMA12']].plot(label='rolling mean on passenger cars in use', figsize=(16, 8))
plt.title("Plot of Sqrt transformed Differenced Time Series vs Moving Average(k=12 months) Sqrt transformed Differenced Time series data")
plt.show()
plt.title("Plot of Moving Average(k=12) Differenced Logarithmic Time Series ")
plt.plot(df['DiffLOGMA12'])
plt.show()
plt.title("Plot of Moving Average(k=12) Differenced Square Root Time Series ")
plt.plot(df['DiffSRMA12'])
plt.show()
pd.plotting.autocorrelation_plot(df.dropna()['DiffMA12'])
plt.show()
pd.plotting.autocorrelation_plot(df.dropna()['DiffLOGMA12'])
plt.title("Plot of Moving Average(k=12) Logarithmic Time Differenced Series ")
plt.show()
pd.plotting.autocorrelation_plot(df.dropna()['DiffSRMA12'])
plt.show()

#################################################################

result = adfuller(df.dropna()['DiffMA12'])
print("Fuller test on Moving Average(k=12) dataset:")
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
result = adfuller(df.dropna()['DiffLOGMA12'])
print("Fuller test on Moving Average of Logarithmic time series dataset:")
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
result = adfuller(df.dropna()['DiffSRMA12'])
print("Fuller test on Moving Average of Square root time series dataset: dataset:")
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))