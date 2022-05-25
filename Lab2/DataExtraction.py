from pandas_datareader import data as pdr
from matplotlib import pyplot as plt
import yfinance as yf
import pandas as pd
yf.pdr_override() # <== that's all it takes :-)

# download dataframe
dataTCS = pdr.get_data_yahoo("TCS.NS", start="2017-05-25", end="2022-05-25")
dataASIANPAINT = pdr.get_data_yahoo("ASIANPAINT.NS", start="2017-05-25", end="2022-05-25")
dataBRITANNIA = pdr.get_data_yahoo("BRITANNIA.NS", start="2017-05-25", end="2022-05-25")
plt.plot(dataTCS['Close'])
plt.plot(dataASIANPAINT['Close'])
plt.plot(dataBRITANNIA['Close'])
#plt.show()
# dictionary of lists 
dict = {'TCS': dataTCS['Close'], 'ASIANPAINT': dataASIANPAINT['Close'], 'BRITANNIA': dataBRITANNIA['Close']} 
     
df = pd.DataFrame(dict)
print(df)
  
# saving the dataframe
df.to_csv('StocksData.csv')
