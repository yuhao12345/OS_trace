import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from itertools import product
from sklearn.metrics import classification_report
import time
import sys

start = time.time()
#%% load data
input_feature=73

train_input_path = sys.argv[1]
# train_input_path="E:/github/IONet-Models/datasets/azure-A.csv"
print("Train input path: " + train_input_path)

train_data = pd.read_csv(train_input_path, dtype='float32',sep=',', header=None)

# frac=1
train_data = train_data.sample(frac=1).reset_index(drop=True)
train_data = train_data.values
# 31 = number of input features
train_input = train_data[:,:input_feature]
train_output = train_data[:,input_feature]

lat_threshold = np.percentile(train_output, 85)

print("lat_threshold: ",lat_threshold)
num_train_entries = int(len(train_output) * 0.80)
print("num train entries: ",num_train_entries)

train_Xtrn = train_input[:num_train_entries,:]
train_Xtst = train_input[num_train_entries:,:]
train_Xtrn = np.array(train_Xtrn)
train_Xtst = np.array(train_Xtst)

#Classification
train_y = []
for num in train_output:
    labels = [0] * 2
    if num < lat_threshold:
        labels[0] = 1
    else:
        labels[1] = 1
    train_y.append(labels)


#print(y)
train_ytrn = train_y[:num_train_entries]
train_ytst = train_y[num_train_entries:]
train_ytrn = np.array(train_ytrn).astype('float32')
train_ytst = np.array(train_ytst).astype('float32')

#%%
#-------------------------Print FP TP FN TN--------------------------
def perf_measure(y_actual, y_pred):
    class_id = set(y_actual).union(set(y_pred))
    TP = []
    FP = []
    TN = []
    FN = []

    for index ,_id in enumerate(class_id):
        TP.append(0)
        FP.append(0)
        TN.append(0)
        FN.append(0)
        for i in range(len(y_pred)):
            if y_actual[i] == y_pred[i] == _id:
                TP[index] += 1
            if y_pred[i] == _id and y_actual[i] != y_pred[i]:
                FP[index] += 1
            if y_actual[i] == y_pred[i] != _id:
                TN[index] += 1
            if y_pred[i] != _id and y_actual[i] != y_pred[i]:
                FN[index] += 1
    total_data = len(y_actual)
    print ("total dataset " + str(total_data))
    print ( "  id  TP  FP  TN   FN")
    for x ,_id in enumerate(class_id):
        print ("  " + str(_id) + "\t" + str(TP[x]) + "\t" +  str(FP[x]) + "\t" +  str(TN[x]) + "\t" +  str(FN[x]))
    
    print ("\n_id    %FP        %FN")
    percentFP = []
    percentFN = []
    for x ,_id in enumerate(class_id):
        if ( FN[x]+ TP[x] > 0 and FP[x]+ TN[x] > 0):
            percentFP.append(FP[x]/( FP[x]+ TN[x])*100)
            percentFN.append(FN[x]/( FN[x]+ TP[x])*100)
            print ("  " + str(_id) + "   " + str(float("{:.2f}".format(percentFP[x]))) + " \t\t " + str(float("{:.2f}".format(percentFN[x]))))
  
    # print (  "\nmacro %FP and %FN = " + str(float("{:.2f}".format(np.sum(percentFP)/2))))
    print ("\n")
#%%
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


#%%

model = tf.keras.Sequential()
# input layer, feature = 31 (input_dim), hidden layer with 256 neurons
# ModelB feature = 24
# ModelA feature = 17, 256->128
# ModelC, model.add(Dense(256)), R=4, feature = 31
# ModelD, input_dim = 73, model.add(dense(512)); model.add(dense(256))

model.add(layers.Dense(256, input_dim=input_feature, activation='sigmoid'))
model.add(layers.Dense(512, activation='sigmoid'))
model.add(layers.Dense(256, activation='sigmoid'))
model.add(layers.Dense(2, activation='linear'))#,kernel_regularizer=regularizers.l2(0.001)))

model.compile(optimizer='adam', loss=w_categorical_crossentropy, metrics=['accuracy'])

# model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# model.compile(optimizer='adam', loss='categorical_hinge', metrics=['accuracy'])

#%%
for i in range(50):
    model.fit(train_Xtrn, train_ytrn, epochs=1, batch_size=128, verbose=0)
    print('Iteration '+str(i)+'\n')
    print('On test dataset:\n')
    train_Y_test = np.argmax(train_ytst, axis=1) # Convert one-hot to index
    # train_y_pred = model.predict_classes(train_Xtst)
    train_y_pred = np.argmax(model.predict(train_Xtst),axis=1)
    
    print(classification_report(train_Y_test, train_y_pred, digits=4))
    perf_measure(train_Y_test, train_y_pred)

    count = 0
    for layer in model.layers:
        weights = layer.get_weights()[0]
        biases = layer.get_weights()[1]
        name = train_input_path +'.weight_' + str(count) + '.csv' #output_dir_path+'/SSD_31col_iteration_'+str(i)+'_weightcustom1_' + str(count) + '.csv'
        name_b = train_input_path + '.bias_' + str(count) + '.csv' #output_dir_path+'/SSD_31col_iteration_'+str(i)+'_biascustom1_' + str(count) + '.csv'
        np.savetxt(name, weights, delimiter=',')
        np.savetxt(name_b, biases, delimiter=',')
        count += 1

end = time.time()
print("Training time = " + str(np.float16(end - start)) + " s")