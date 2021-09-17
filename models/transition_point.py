import numpy as np
import pandas as pd

# from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt



deviceName = ['nvme0n1', 'nvme1n1', 'nvme2n1', 'nvme3n1', 'sdd', 'sde']
traceType  = ['azure', 'bingI', 'bingS', 'cosmos']
editoption = ['', 'out-rerated-10.0', 'out-rerated-100.0']

path_pre='E:/github/IONet-Models/TRACEPROFILE/'
path_lat='.trace_sub-A.tmp'
#%%  extract raw dist of latency

i=0
t=1
j=2

fileInfo=deviceName[i]+'_'+traceType[t]+'_'+ editoption[j]
train_input_path=path_pre + deviceName[i] + '/' +editoption[j] +'/'+traceType[t] + path_lat
print("Train input path: " + train_input_path)

df = pd.read_csv(train_input_path, dtype='float32',sep=',', header=None)
df=df.values

lat_df=df[:,-1]

pdf, bins= np.histogram(lat_df,bins=1000, density=True)


# plt.figure()
# plt.plot(bins[:-1], pdf,'.')
# plt.xlabel('latency')
# plt.ylabel('pdf')

dbins=bins[1]-bins[0]
cdf= np.cumsum(pdf)*dbins
# plt.figure()
# plt.plot(bins[:-1],cdf)
# plt.xlabel('latency')
# plt.ylabel('cdf')

#%% smooth cdf

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
    



X = bins[0:-1]+dbins/2
y = cdf


lw_reg=LWRegressor(1/100)
lw_reg.fit(X, y)
predictions_lw = lw_reg.predict(X.reshape(-1,1))

# plt.figure()
# plt.scatter(X, y, s=10, color='b', alpha=0.5, label='data')
# plt.plot(X, predictions_lw, color='k', label='locally weighted regression')
# plt.legend()

#%%   curvature

lat_p95 = np.percentile(lat_df, 95)
ratio=0.95/lat_p95

x=bins[0:-1]+dbins/2

y = predictions_lw/ratio
# y = cdf/ratio
# cdf_unique, cdf_index = np.unique(predictions_lw, return_index=True)
# x_unique=x[cdf_index]

cdf_1d=(y[1:]-y[0:-1])/dbins

cdf_2d=(cdf_1d[1:]-cdf_1d[0:-1])/dbins
# cdf_2d_avg = pd.Series(cdf_2d).rolling(14).mean().to_numpy()
k=cdf_2d/(1+cdf_1d[0:-1]**2)**1.5

#%%

lat_p70 = np.percentile(lat_df, 70)
s_index = np.searchsorted(x, lat_p70)
index=np.argmin(k[s_index:]) + s_index

# index=88

plt.figure()
# plt.rcParams['font.size'] = '16'

plt.subplot(2,1,1)
plt.plot(x,y*ratio,'.')
plt.ylabel('CDF')
plt.title(fileInfo)
plt.plot(x[index],y[index]*ratio,'o')
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
# plt.subplot(4,1,2)
# plt.plot(x[0:-1], cdf_1d,'.')

# plt.subplot(4,1,3)
# plt.plot(x[0:-2],cdf_2d,'.')

plt.subplot(2,1,2)
plt.plot(x[0:-2],k,'.')
plt.plot(x[index],k[index],'o',markerfacecolor='none')
plt.ylabel('curvature')
plt.xlabel('latency')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

# plt.savefig("C:/Users/kymws/Desktop/practicum/curvature/"+fileInfo.replace('/', '_')+".png")

print('p_transition:'+str(x[index])+"  "+str(y[index][0]*ratio))

