import numpy as np
import matplotlib.pyplot as plt

n=np.array([2, 3, 4, 10])
n=2*n+1

accu_DT=[0.962156, 0.9642458, 0.9660614, 0.963257]
accu_A=[0.955467, 0.954618, 0.9527998, 0.94828164]
accu_D=[0.9565116, 0.95454545, 0.95655355, 0.9585694]

plt.figure()
plt.plot(n,accu_DT,'-o')
plt.plot(n,accu_A,'-o')
plt.plot(n,accu_D,'-o')
plt.legend(['DT','modelA','modelD'])
plt.xlabel("# of features")
plt.ylabel("Accuracy")
plt.title('nvme0n1/out-resized-100.0/azure')

#%%
n=np.array([2, 3, 4, 10])
n=2*n+1

accu_DT=[0.9313818, 0.93368, 0.933118, 0.932408]
accu_A=[0.921502, 0.9225651, 0.92279240, 0.9206844]
accu_D=[0.91347019,0.9181811,0.9144352,0.916951]

plt.figure()
plt.plot(n,accu_DT,'-o')
plt.plot(n,accu_A,'-o')
plt.plot(n,accu_D,'-o')
plt.legend(['DT','modelA','modelD'])
plt.xlabel("# of features")
plt.ylabel("Accuracy")
plt.title('nvme0n1/out-resized-100.0/bingI')

#%% accu, FP vs # of NN layers

fileInfo='nvme0n1/bingI'
layer=np.arange(1,6)
accuracy=np.array([0.9197,0.924279,0.922870,0.918373,0.91822443598])
FP=np.array([0.0493,0.04485,0.04753,0.054353,0.050395])

fileInfo='nvme0n1/out-rerated-10.0/bingI'
layer=np.arange(1,6)
accuracy=np.array([0.8843949,0.8849985,0.877077,0.872658,0.876980])
FP=np.array([0.076124,0.072795,0.082879,0.089903,0.080469])

fileInfo='nvme0n1/out-rerated-100.0/bingI'
layer=np.arange(1,6)
accuracy=np.array([0.899478,0.891886,0.891046,0.884205,0.8906338])
FP=np.array([0.0443,0.05673,0.05959,0.06942,0.061868])

fileInfo='nvme0n1/bingS'
layer=np.arange(1,5)
accuracy=np.array([0.7955,0.801410,0.80426,0.797119])
FP=np.array([0.12297,0.1135,0.1057,0.119767])
#%%  accu, FP vs # of NN layers

import matplotlib.ticker as mticker
fileInfo='nvme1n1/bingI'
layer=np.arange(1,6)
accuracy=np.array([0.9237,0.91975,0.91431,0.92262,0.9169])
FP=np.array([0.04459,0.04656,0.04579,0.04243,0.0423])

plt.figure().suptitle(fileInfo + "  IP=0.915")

plt.subplot(1,2,1)
plt.ylabel('Accuracy')
plt.xlabel('# layer')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.plot(layer,accuracy)

plt.subplot(1,2,2)
plt.plot(layer,FP)
plt.ylabel('FP')
plt.xlabel('# layer')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))

#%% accu, FP vs # of input features, method
fileInfo='nvme0n1/out-resized-100.0/azure'
n=np.array([2,3,4,10])
accu_NN=np.array([0.95827,0.95483,0.9545524,0.9549])
FP_NN=np.array([0.025422, 0.02688,0.027579,0.02725])

plt.figure().suptitle(fileInfo + "  IP=0.8968")

plt.subplot(1,2,1)
plt.ylabel('Accuracy')
plt.xlabel('# feature n')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.plot(n,accu_NN)
plt.plot(n_DT[1:], accu_DT[1:])
plt.legend(['NN','DT'])

plt.subplot(1,2,2)
plt.plot(n,FP_NN)
plt.plot(n_DT[1:], FP_DT[1:])
plt.ylabel('FP')
plt.xlabel('# feature n')
plt.legend(['NN','DT'])
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))

#%% accu, FP vs # of input features, method
fileInfo='nvme0n1/out-resized-100.0/bingI'
n=np.array([2,3,4,10])
accu_NN=np.array([0.957,0.957986,0.95558,0.9534])
FP_NN=np.array([0.0145,0.01678,0.01537,0.01646])

plt.figure().suptitle(fileInfo + "  IP=0.8968")

plt.subplot(1,2,1)
plt.ylabel('Accuracy')
plt.xlabel('# feature n')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.plot(n,accu_NN)
plt.plot(n_DT[1:], accu_DT[1:])
plt.legend(['NN','DT'])

plt.subplot(1,2,2)
plt.plot(n,FP_NN)
plt.plot(n_DT[1:], FP_DT[1:])
plt.ylabel('FP')
plt.xlabel('# feature n')
plt.legend(['NN','DT'])
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))

#%% accu, FP vs # of input features, method
fileInfo='nvme1n1/bingI'
n=np.array([2,3,4,10])
accu_NN=np.array([0.9209,0.9218,0.92124,0.919430])
FP_NN=np.array([0.04486,0.04374,0.042655,0.045548])

plt.figure().suptitle(fileInfo + "  IP= 0.915")

plt.subplot(1,2,1)
plt.ylabel('Accuracy')
plt.xlabel('# feature n')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.plot(n,accu_NN)
plt.plot(n_DT[1:], accu_DT[1:])
plt.legend(['NN','DT'])

plt.subplot(1,2,2)
plt.plot(n,FP_NN)
plt.plot(n_DT[1:], FP_DT[1:])
plt.ylabel('FP')
plt.xlabel('# feature n')
plt.legend(['NN','DT'])
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))