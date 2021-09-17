import numpy as np
import pandas as pd

# 'nvme0n1/azure'
# 'nvme0n1/out-resized-10.0/azure'
# 'nvme0n1/out-resized-100.0/azure'
# 'nvme0n1/out-rerated-10.0/azure'
# 'nvme0n1/out-rerated-100.0/azure'

# 'nvme0n1/bingI'
# 'nvme0n1/out-resized-10.0/bingI'
# 'nvme0n1/out-resized-100.0/bingI'
# 'nvme0n1/out-rerated-10.0/bingI'
# 'nvme0n1/out-rerated-100.0/bingI'

# 'nvme0n1/bingS'
# 'nvme0n1/out-resized-10.0/bingS'
# 'nvme0n1/out-resized-100.0/bingS'
# 'nvme0n1/out-rerated-10.0/bingS'
# 'nvme0n1/out-rerated-100.0/bingS'

fileInfo_list=['nvme1n1/bingS','nvme1n1/out-rerated-10.0/bingS','nvme1n1/out-rerated-100.0/bingS']
corr=[]
for fileInfo in fileInfo_list:
    train_input_path="E:/github/IONet-Models/TRACEPROFILE/"+fileInfo+".trace_sub-A.tmp"
    
    df = pd.read_csv(train_input_path, dtype='float32',sep=',', header=None)
    df=df.values
    corr.append(np.corrcoef(df[:,2],df[:,5])[0,1])
    
corr_bingS=np.array(corr)

plt.plot(corr_azure)
plt.plot(corr_bingI)
plt.plot(corr_bingS)
plt.legend(['azure','bingI','bingS'])
plt.ylabel('correlation')
