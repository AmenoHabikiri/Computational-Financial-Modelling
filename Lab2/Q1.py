import pandas as pd
import math
from matplotlib import pyplot as plt
from numpy import log
import numpy as np
data = pd.read_csv('StocksData.csv')
data[['TCS','ASIANPAINT','BRITANNIA']].plot(label='ClosePreice over the years', figsize=(16, 8))
#plt.show()
data[['LOGTCS','LOGASIANPAINT','LOGBRITANNIA']] = data[['TCS','ASIANPAINT','BRITANNIA']].pct_change().apply(lambda x: np.log(1+x))
SD=[0,0,0]
print(data)
SD[0]=np.sqrt((data['LOGTCS'].var())*250)
SD[1]=np.sqrt((data['LOGASIANPAINT'].var())*250)
SD[2]=np.sqrt((data['LOGBRITANNIA'].var())*250)
print(SD)
