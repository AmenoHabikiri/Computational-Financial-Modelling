import pandas as pd
import math
from matplotlib import pyplot as plt
from numpy import log
import numpy as np
data = pd.read_csv('StocksData.csv')
data[['TCS','ASIANPAINT','BRITANNIA']].plot(label='ClosePreice over the years', figsize=(16, 8))
#plt.show()
data1=data
data1 = data[['TCS','ASIANPAINT','BRITANNIA']].pct_change().apply(lambda x: np.log(1+x))
cov_matrix=data[['TCS','ASIANPAINT','BRITANNIA']].cov()
print(data1)
print(cov_matrix)

