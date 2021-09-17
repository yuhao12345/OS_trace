"""curvature-based transition point finder
"""
# Author: Yuhao Kang  <yuhaok@uchicago.edu>

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn


def rbf(l,x,gamma):
    # l is array, x is point
    return np.exp(-abs(l-x)**2*gamma)


class LWRegressor(sklearn.base.RegressorMixin):
    
    def __init__(self, gamma):
        self.gamma = gamma
        super().__init__()
        
    def fit(self, X,y):
        self.X = X
        self.y = y
        
    def predict_onepoint(self, target):
        z1=np.ones(len(self.X))
        z2=self.X
        z=np.array([z1,z2]).T
        w = np.diag(rbf(self.X, target, self.gamma))
        beta=np.linalg.lstsq(z.T@w@z, z.T@w@self.y, rcond=None)[0]
        return beta[0]+beta[1]*target
   
    def predict(self, T):
        predictions = []

        for i in range(T.shape[0]):
            predictions.append(self.predict_onepoint(T[i]))
        return np.asarray(predictions)
    
deviceName = ['nvme0n1', 'nvme1n1', 'nvme2n1', 'nvme3n1', 'sdd', 'sde']
traceType  = ['azure', 'bingI', 'bingS', 'cosmos']
editoption = ['', 'out-rerated-10.0', 'out-rerated-100.0']

path_pre='E:/github/IONet-Models/TRACEPROFILE/'
path_lat='.trace_sub-A.tmp'
#  extract raw dist of latency

def get_transition_point(i=0,j=0,t=0):
    fileInfo=deviceName[i]+'_'+traceType[t]+'_'+ editoption[j]
    train_input_path=path_pre + deviceName[i] + '/' +editoption[j] +'/'+traceType[t] + path_lat
    print("Train input path: " + train_input_path)
    
    df = pd.read_csv(train_input_path, dtype='float32',sep=',', header=None)
    df=df.values
    
    lat_df=df[:,-1]
    
    pdf, bins= np.histogram(lat_df,bins=1000, density=True)
    
    dbins=bins[1]-bins[0]
    cdf= np.cumsum(pdf)*dbins
    
    # smooth cdf
    
    X = bins[0:-1]+dbins/2
    y = cdf
    
    
    lw_reg=LWRegressor(1/100)
    lw_reg.fit(X, y)
    predictions_lw = lw_reg.predict(X.reshape(-1,1))
    
    # curvature
    
    lat_p95 = np.percentile(lat_df, 95)
    ratio=0.95/lat_p95
    
    x=bins[0:-1]+dbins/2
    
    y = predictions_lw/ratio

    
    cdf_1d=(y[1:]-y[0:-1])/dbins
    
    cdf_2d=(cdf_1d[1:]-cdf_1d[0:-1])/dbins
    k=cdf_2d/(1+cdf_1d[0:-1]**2)**1.5

    
    lat_p70 = np.percentile(lat_df, 70)
    s_index = np.searchsorted(x, lat_p70)
    index=np.argmin(k[s_index:]) + s_index
    
# =============================================================================
#     plt.figure()
#     # plt.rcParams['font.size'] = '16'
#     plt.ion()
#     plt.subplot(2,1,1)
#     plt.plot(x,y*ratio,'.')
#     plt.ylabel('CDF')
#     plt.title(fileInfo)
#     plt.plot(x[index],y[index]*ratio,'o')
#     plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
#     plt.subplot(2,1,2)
#     plt.plot(x[0:-2],k,'.')
#     plt.plot(x[index],k[index],'o',markerfacecolor='none')
#     plt.ylabel('curvature')
#     plt.xlabel('latency')
#     plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
#     plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
#     plt.ioff()
#     plt.savefig('E:/github/IONet-Models/t0_transition/'+deviceName[i] + '_' +editoption[j] +'_'+traceType[t]+'.png')
#     plt.close()
# =============================================================================
    
    return [x[index], y[index][0]*ratio]

#%%
dict_transition={}
for i in range(len(deviceName)):
    for t in range(len(traceType)):
        for j in range(len(editoption)):
            dict_transition[deviceName[i]+'_'+traceType[t]+'_'+ editoption[j]] = get_transition_point(i,j,t)

#%%  save dict to csv
import csv

with open('E:/github/IONet-Models/t0_transition/dict_transition.csv', "w") as f:
    writer = csv.writer(f)
    for i in dict_transition:
      writer.writerow([i, dict_transition[i][0],dict_transition[i][1]])
f.close() 

