import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
# import time
import csv

# define loss function

custom_loss = 5.0
w_array = np.ones((2,2))
w_array[1, 0] = custom_loss   #Custom Loss Multiplier
#w_array[0, 1] = 1.2

def w_categorical_crossentropy(y_true, y_pred):
    weights = w_array
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0],dtype=K.floatx())
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
    y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max), K.floatx())
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
    cross_ent = tf.keras.losses.categorical_hinge(y_true,y_pred) #
    cross_ent = tf.dtypes.cast(cross_ent, K.floatx())
    # cross_ent = K.categorical_crossentropy(y_true,y_pred, from_logits=False)
    return cross_ent * final_mask

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
path_lat='-D.ionet-dataset'


def loadData(i,t,j,n_feature,n_NNlayes):
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
    
    train = train[:,np.concatenate((range(30-3*n_feature,33),range(73-4*n_feature,74)))]
    test = test[:,np.concatenate((range(30-3*n_feature,33),range(73-4*n_feature,74)))]
    
    #Classification
    def classify(output, lat_threshold):
        train_y = []
        for num in output:
            labels = [0] * 2
            if num < lat_threshold:
                labels[0] = 1
            else:
                labels[1] = 1
            train_y.append(labels)
        return np.array(train_y).astype('float32')
    
    train_y = classify(train[:,-1], lat_threshold)
    test_y = classify(test[:,-1], lat_threshold)
    
    train_X = train[:,:-1]
    test_X = test[:,:-1]
    return train_X,train_y,test_X,test_y


def nnlayer(n=1):
    model = tf.keras.Sequential()
    if n==1:
        
        model.add(layers.Dense(128,  activation='relu'))
        model.add(layers.Dense(2, activation='linear'))
        
        model.compile(optimizer='adam', loss=w_categorical_crossentropy, metrics=['accuracy'])
        return model
    
    if n==2:
        model.add(layers.Dense(128,  activation='relu'))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(2, activation='linear'))
        
        model.compile(optimizer='adam', loss=w_categorical_crossentropy, metrics=['accuracy'])
        return model
    
    if n==3:
        model.add(layers.Dense(128,  activation='relu'))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(2, activation='linear'))
        
        model.compile(optimizer='adam', loss=w_categorical_crossentropy, metrics=['accuracy'])
        return model
    
    if n==4:
        model.add(layers.Dense(128,  activation='relu'))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(2, activation='linear'))
        
        model.compile(optimizer='adam', loss=w_categorical_crossentropy, metrics=['accuracy'])
        return model
    
    if n==5:
        model.add(layers.Dense(128,  activation='relu'))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(2, activation='linear'))
        
        model.compile(optimizer='adam', loss=w_categorical_crossentropy, metrics=['accuracy'])
        return model



#  slow: 1   quick :0 
def accu_FP(i,t,j,n_feature,n_NNlayes):
    train_X,train_y,test_X,test_y=loadData(i,t,j,n_feature,n_NNlayes)
    num_test = 5
    accu_list=np.zeros(num_test)
    FP_list=np.zeros(num_test)
    for i in range(num_test):
        model=nnlayer(n_NNlayes)
        model.fit(train_X, train_y, epochs=5, batch_size=128, verbose=0)
        # print('Iteration '+str(i))
    
        train_Y_test = np.argmax(test_y, axis=1) # Convert one-hot to index
        train_y_pred = np.argmax(model.predict(test_X),axis=1)
        
        accu_list[i]=sum(train_Y_test==train_y_pred)/len(train_Y_test)
        FP_list[i]=sum((train_Y_test-train_y_pred)==1)/len(train_Y_test)
    
    
    
    return np.mean(accu_list), np.mean(FP_list)


# deviceName = ['nvme0n1', 'nvme1n1', 'nvme2n1', 'nvme3n1', 'sdd', 'sde']
# traceType  = ['azure', 'bingI', 'bingS', 'cosmos']
# editoption = ['', 'out-rerated-10.0', 'out-rerated-100.0']

i=4
t=2
j=0
# n_feature = 1   # so # of features=2*n+1
# n_NNlayes = 2

fileInfo = deviceName[i]+'_'+traceType[t]+'_'+ editoption[j]

# dict_accu_FP={}
# dict_accu_FP[fileInfo] = accu_FP(i,t,j,n_feature,n_NNlayes)
n_feature_list=np.arange(1,8,2)
n_NNlayes_list=np.arange(1,6,1)

accu_matrix=np.zeros([len(n_feature_list),len(n_NNlayes_list)])
FP_matrix=np.zeros([len(n_feature_list),len(n_NNlayes_list)])

for x in range(len(n_feature_list)):
    for y in range(len(n_NNlayes_list)):
        print(str(x)+' '+str(y))
        n_feature=n_feature_list[x]
        n_NNlayes=n_NNlayes_list[y]
        accu_matrix[x,y],FP_matrix[x,y]=accu_FP(i,t,j,n_feature,n_NNlayes)
        

np.savetxt("E:/github/IONet-Models/accu_FP/"+fileInfo+"_accu.csv", accu_matrix, delimiter=",")
np.savetxt("E:/github/IONet-Models/accu_FP/"+fileInfo+"_FP.csv", FP_matrix, delimiter=",")

plt.figure()
plt.suptitle(fileInfo+'  '+'p0='+str(round(dic[fileInfo][1],3)))

plt.subplot(1,2,1)
plt.plot(n_feature_list,accu_matrix,'o-')
plt.xlabel('n_feature')
plt.ylabel('Accu')
plt.legend(n_NNlayes_list)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.subplot(1,2,2)
plt.plot(n_feature_list,FP_matrix,'o-')
plt.xlabel('n_feature')
plt.ylabel('FP')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

