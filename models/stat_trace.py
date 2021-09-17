import numpy as np
import matplotlib.pyplot as plt

#%%
deviceName = ['nvme0n1', 'nvme1n1', 'nvme2n1', 'nvme3n1', 'sdd', 'sde']
traceType  = ['azure', 'bingI', 'bingS', 'cosmos']

for i in range(6):
    path='E:/github/IONet-Models/TRACEPROFILE/'+deviceName[i]+'/bingI-A.ionet-dataset'
    df=np.loadtxt(path,delimiter=',')
    
    lat=df[:,-1]   # latency
    
    # getting data of the histogram
    count, bins_count = np.histogram(lat, bins=500)
      
    # finding the PDF of the histogram using count values
    pdf = count / sum(count)
      
    # using numpy np.cumsum to calculate the CDF
    # We can also find using the PDF values by looping and adding
    cdf = np.cumsum(pdf)
      
    # plotting PDF and CDF
    # plt.plot(bins_count[1:], pdf, color="red", label="PDF")
    plt.plot(bins_count[1:], cdf, label="CDF")
plt.xlim(0,3000)
plt.ylabel("CDF")
plt.xlabel("Latency (microSec)")
plt.legend(deviceName)
plt.title("traceType: bingI")


#%%

path="E:\github\IONet-Models\scripts\data-to-plot\\traceType=azure\deviceName=nvme0n1\modelID=model-C\editOption-percentFPR.txt"

label = np.loadtxt(path, delimiter=',', dtype=np.str, usecols=[0]).tolist()
value = np.loadtxt(path, delimiter=',', usecols=[1]).tolist()

plt.figure()
# ax = fig.add_axes([0,0,1,1])
plt.bar(label, value)
# plt.xticks(rotation=45)
plt.ylabel("FPR") #Accuracy FPR
plt.title("traceType=azure\deviceName=nvme0n1\modelID=model-C")
plt.show()

#%%
# import pandas as pd

path="E:\github\IONet-Models\scripts\data-to-plot\deviceName=nvme0n1\modelID=model-A\\traceType-editOption-accuracy.txt"

traceType = np.loadtxt(path, delimiter=',', dtype=np.str, usecols=[0])
edit = np.loadtxt(path, delimiter=',', dtype=np.str, usecols=[1])
accu = np.loadtxt(path, delimiter=',', usecols=[2])

traceType  = ['azure', 'bingI', 'bingS', 'cosmos']
rerated100=[86.21,83.41,84.91,85.02]
resize100=[93.88,92.3,79.47999999999999,0]
ori=[82.42,90.72,83.74000000000001,98.93]

x = np.arange(len(traceType))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, rerated100, width/2, label='rerated-100x')
rects2 = ax.bar(x, resize100, width/2, label='resized-100x')
rects3 = ax.bar(x + width/2, ori, width/2, label='original')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
# ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(traceType)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.bar_label(rects3, padding=3)
fig.tight_layout()