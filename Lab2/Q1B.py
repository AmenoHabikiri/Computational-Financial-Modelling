from audioop import avg
import pandas as pd
import math
from matplotlib import pyplot as plt
from numpy import log
import numpy as np
import random
data = pd.read_csv('StocksData.csv')
data[['TCS','ASIANPAINT','BRITANNIA']].plot(label='ClosePreice over the years', figsize=(16, 8))
#plt.show()
data1=data
data1 = data[['TCS','ASIANPAINT','BRITANNIA']].pct_change().apply(lambda x: np.log(1+x))
cov_matrix=data1[['TCS','ASIANPAINT','BRITANNIA']].cov()
corr_matrix=data1[['TCS','ASIANPAINT','BRITANNIA']].corr()
#print(data1)
#print(cov_matrix)
#print(corr_matrix)
#randomly weighted portfolio's variance
#to calculate yearly return assuming 250 market days per year
k1=[] # Define an empty array for TCS
k2=[] # Define an empty array for ASIANPAINT
k3=[] #Define an empty array for BRITANNIA
for i in range(250,len(data['Date'])):
    k1.append((data['TCS'][i]-data['TCS'][i-250])/data['TCS'][i-250])
    k2.append((data['ASIANPAINT'][i]-data['ASIANPAINT'][i-250])/data['ASIANPAINT'][i-250])
    k3.append((data['BRITANNIA'][i]-data['BRITANNIA'][i-250])/data['BRITANNIA'][i-250])
ind_er=[]
ind_er.append(sum(k1)/len(k1))
ind_er.append(sum(k2)/len(k2))
ind_er.append(sum(k3)/len(k3))
print(ind_er)
############################3
p_ret = [] # Define an empty array for portfolio returns
p_vol = [] # Define an empty array for portfolio volatility
p_weights = [] # Define an empty array for asset weights

num_assets = 3
num_portfolios = 10000
DATA=[]
for portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights = weights/np.sum(weights)
    p_weights.append(weights)
    returns = np.dot(weights, ind_er) # Returns are the product of individual expected returns of asset and its 
                                      # weights 
    p_ret.append(returns)
    var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()# Portfolio Variance
    sd = np.sqrt(var) #Daily standard deviation
    ann_sd = sd*np.sqrt(250) #Annual standard deviation = volatility
    p_vol.append(ann_sd)
DATA=[]
for i in range(len(p_weights)):
    k=[]
    k.append(p_ret[i])
    k.append(p_vol[i])
    for j in range(len(p_weights[i])):
        k.append(p_weights[i][j])
    DATA.append(k)

df = pd.DataFrame(DATA, columns =['Returns', 'Volatility', 'TCSW','ASIANPAINTW','BRITANNIAW'], dtype = float)

plt.subplots(figsize=[10,10])
plt.xlabel("Risk")
plt.ylabel("Return")
plt.scatter(df['Volatility'],df['Returns'])
plt.savefig("img1.jpg")
plt.show()
for i in range(3):
    min_returns=np.min(df["Returns"].values)
    max_returns=np.max(df["Returns"].values)
    val=random.random()*(max_returns-min_returns)+min_returns
    df_filtered=df[df["Returns"]>val]
    plt.subplots(figsize=[10,10])
    plt.xlabel("Risk")
    plt.ylabel("Return")
    min_vol_port = df.iloc[df_filtered['Volatility'].idxmin()]
    risk__=0
    for i in range(len(df["Returns"])):
        if (df["Returns"][i]==min(df_filtered["Returns"])):
            risk__=df["Volatility"][i]
    print(" Plot for return value x=",(df_filtered["Returns"]*100))
    print("  And Minimum Risk value : ",risk__)
    plt.scatter(df_filtered['Volatility'], df_filtered['Returns'],marker='o', s=10, alpha=0.3)
    plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500)
    plt.savefig(f"img{i+2}.jpg") 
    plt.show()
min_vol_port = df.iloc[df['Volatility'].idxmin()]# idxmin() gives us the minimum value in the column specified. 
#plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500)

plt.subplots(figsize=[10,10])
plt.scatter(df['Volatility'], df['Returns'],marker='o', s=10, alpha=0.3)
plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500)
plt.savefig("img2.jpg")
plt.show()
def single_objective_optimization(ret,risk):
    lam=[]
    for i in range(20):
        lam.append(np.random.random())
    n=len(ret)
    n1=len(lam)
    cost_fun=[]
    lam1=[]
    for i in range(n1):
        price_fun=[]
        for j in range(n):
            price_fun.append(lam[i]*ret[j]-(1-lam[i])*risk[j])
        cost_fun.append(max(price_fun))
        lam1.append(lam[i])
    plt.scatter(lam1,cost_fun)
    plt.show()
single_objective_optimization(list(df['Returns']),list(df['Volatility']))
ret_rate=[]
for i in range(3):
    ret_rate.append(np.random.random())





