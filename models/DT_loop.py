import numpy as np
import pandas as pd
# from sklearn.impute import SimpleImputer
# from sklearn.model_selection import validation_curve
# from sklearn.model_selection import learning_curve
from sklearn import tree
# from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
# from sklearn import metrics
import csv

        
# quick: 1   slow:0
def classify(lat_list, lat_threshold):
    return np.array(lat_list<lat_threshold).astype(int)

# quick: 1   slow:0
def accu_fp(test_y, pred_y):
    tmp=test_y==pred_y
    accu=sum(tmp)/len(tmp)
    FP=sum((test_y-pred_y)==-1)/len(tmp)
    return accu,FP

# load transition point of each sample to dictionary
with open('E:/github/IONet-Models/t0_transition/dict_transition.csv',newline='\n') as f:
    reader = csv.reader(f)
    data = list(reader)

dic = {}
for item in data:
    dic[item[0]]=[float(item[1]),float(item[2])]
    

deviceName = ['nvme0n1', 'nvme1n1', 'nvme2n1', 'nvme3n1', 'sdd', 'sde']
traceType  = ['azure', 'bingI', 'bingS', 'cosmos']
editoption = ['', 'out-rerated-10.0', 'out-rerated-100.0']

path_pre='E:/github/IONet-Models/TRACEPROFILE/'
path_lat='.trace_sub-D.tmp'



def DT(i,t,j,n_feature):
    fileInfo=deviceName[i]+'_'+traceType[t]+'_'+ editoption[j]
    train_input_path=path_pre + deviceName[i] + '/' +editoption[j] +'/'+traceType[t] + path_lat
    print("Train input path: " + train_input_path)
    
    lat_threshold, p_threshold= dic[fileInfo]
    
    df = pd.read_csv(train_input_path, dtype='float32',sep=',', header=None)
    
    
    #  split train_data to training and testing set
    
    train=df.sample(frac=0.8,random_state=200) #random state is a seed value
    test=df.drop(train.index)
    
    df=df.values
    train=train.values
    test=test.values
    
    if len(train)>3e4:
        train=train[:30000,:]
    
    
    train_y = classify(train[:,-1], lat_threshold)
    test_y = classify(test[:,-1], lat_threshold)
    
    train_X = train[:,0:-1]
    test_X = test[:,0:-1]
    
    
    input_feature = np.concatenate((range(10-n_feature,11),range(21-n_feature,21)))
    
    # Create Decision Tree classifer object
    clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    
    # Train Decision Tree Classifer
    clf = clf.fit(train_X[:,input_feature],train_y)
    
    #Predict the response for test dataset
    pred_y = clf.predict(test_X[:,input_feature])
    
    
    accu, FP=accu_fp(test_y, pred_y)
    
    return accu, FP

# =============================================================================
# deviceName = ['nvme0n1', 'nvme1n1', 'nvme2n1', 'nvme3n1', 'sdd', 'sde']
# traceType  = ['azure', 'bingI', 'bingS', 'cosmos']
# editoption = ['', 'out-rerated-10.0', 'out-rerated-100.0']
# =============================================================================

i=0
t=1
j=2
fileInfo = deviceName[i]+'_'+traceType[t]+'_'+ editoption[j]

n_feature_list=np.arange(1,8,2)
accu_list=np.zeros(len(n_feature_list))
FP_list=np.zeros(len(n_feature_list))

for x in range(len(n_feature_list)):
    n_feature = n_feature_list[x]
    accu_list[x],FP_list[x] = DT(i,t,j,n_feature)
    
    
plt.figure() 
plt.subplot(1,2,1)
plt.plot(n_feature_list,accu_list,'o-')
plt.xlabel('n_feature')
plt.ylabel('Accu')


plt.subplot(1,2,2)
plt.plot(n_feature_list,FP_list,'o-')
plt.xlabel('n_feature')
plt.ylabel('FP')



accu_matrix= np.loadtxt("E:/github/IONet-Models/accu_FP/"+fileInfo+"_accu.csv",delimiter=",")
FP_matrix = np.loadtxt("E:/github/IONet-Models/accu_FP/"+fileInfo+"_FP.csv", delimiter=",")

plt.suptitle(fileInfo+'  '+'p0='+str(round(dic[fileInfo][1],3)))

n_NNlayes_list=np.arange(1,6,1)

n_feature_list=np.arange(1,10,2)
plt.subplot(1,2,1)

plt.plot(n_feature_list,accu_matrix,'o-')
plt.legend(['DT','NNLayer=1','NNLayer=2','NNLayer=3','NNLayer=4','NNLayer=5'])
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.subplot(1,2,2)
plt.plot(n_feature_list,FP_matrix,'o-')

plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))