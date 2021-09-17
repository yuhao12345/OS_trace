import numpy as np
import matplotlib.pyplot as plt

deviceName = ['nvme0n1', 'nvme1n1', 'nvme2n1', 'nvme3n1', 'sdd', 'sde']
traceType  = ['azure', 'bingI', 'bingS', 'cosmos']
editoption = ['', 'out-rerated-10.0', 'out-rerated-100.0']

path_pre='E:/github/IONet-Models/TRACEPROFILE/'
path_lat='.trace_sub-A.tmp'

corr_matrix=np.zeros([len(deviceName),len(traceType),len(editoption)])


for i in range(6):
    for j in range(3):
        for k in range(4):
            path=path_pre + deviceName[i] + '/' +editoption[j] +'/'+traceType[k] + path_lat
            
            df=np.loadtxt(path,delimiter=',')
            
            l0 = df[:,2]
            
            t0 = df[:,5]
            
            corr = np.corrcoef(l0,t0)[0,1]
            
            corr_matrix[i,k,j]=corr
# =============================================================================
#             random_index = np.random.randint(0, len(l0), size=5000)
#             
#             plt.figure()
#             plt.ion()
#             plt.plot(l0[random_index] , t0[random_index], '.')
#             plt.xlabel('l0')
#             plt.ylabel('t0')
#             plt.ioff()
#             plt.savefig('E:/github/IONet-Models/corr_l0_t0/'+deviceName[i] + '_' +editoption[j] +'_'+traceType[k]+'.png')
#             plt.close()
# =============================================================================

#%%
plt.figure()
for i in range(len(deviceName)):
    plt.subplot(2,3,i+1)
    plt.rc('font', size=14)
    plt.plot(corr_matrix[i,:,:].T,'o-')
    # plt.xlabel('edit option')
    plt.ylabel('corr')
    if i==0:
        plt.legend(traceType)
    plt.title(deviceName[i])
    plt.xticks([0, 1, 2], editoption)
    # plt.savefig('E:/github/IONet-Models/corr_l0_t0/corr_'+deviceName[i] +'.png')
