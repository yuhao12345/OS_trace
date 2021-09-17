import numpy as np
import pandas as pd
# from sklearn.impute import SimpleImputer
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn import tree
# from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import metrics

lat_threshold, p_threshold= 219.0845,  0.9150814921439359

fileInfo='nvme1n1/bingI'

#%%  load raw data
# input_feature=21

# train_input_path = sys.argv[1]  #out-resized-100.0
# train_input_path="E:/github/IONet-Models/TRACEPROFILE/sdd/out-rerated-100.0/azure.trace_sub-A.tmp"

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


train_input_path="E:/github/IONet-Models/TRACEPROFILE/"+fileInfo+".trace_sub-D.tmp"
print("Train input path: " + train_input_path)

df = pd.read_csv(train_input_path, dtype='float32',sep=',', header=None)


#  split train_data to training and testing set

train=df.sample(frac=0.8,random_state=200) #random state is a seed value
test=df.drop(train.index)
# train_2000=train.sample(n=2000)

#  df to array
df=df.values
train=train.values
test=test.values
# train_2000=train_2000.values

#%%
# curvature method


# =============================================================================
# lat_df=df[:,-1]
# lat_p70 = np.percentile(lat_df, 70)
# lat_df_filter=lat_df[lat_df>lat_p70]
# 
# n_total=len(lat_df)
# n_filter=len(lat_df_filter)
# 
# pdf, bins= np.histogram(lat_df_filter,bins=int(len(lat_df_filter)/3), density=True)
# =============================================================================

# =============================================================================
# dbins=bins[1]-bins[0]
# cdf= np.cumsum(pdf)*dbins
# 
# a=np.linspace(0,1,100)
# index_a=np.zeros(100)
# p_a=np.zeros(100)
# j=0
# for i in range(len(cdf)):
#     if cdf[i]>a[j]:
#         p_a[j]=cdf[i]
#         index_a[j]=bins[i]
#         j+=1
# p_a[-1]=1
# index_a[-1]= bins[-1]  
# 
# plt.figure()
# 
# plt.plot(index_a,p_a,'.')   
# 
# plt.figure()
# pdf=(p_a[1:]-p_a[:-1])/(index_a[1:]-index_a[:-1])
# plt.plot(index_a[:-1],pdf,'.')   
# =============================================================================

# =============================================================================
# pdf_avg = pd.Series(pdf).rolling(10).mean().to_numpy()
# 
# plt.figure()
# plt.plot(bins[:-1], pdf,'.')
# # plt.plot(pdf_avg,'.')
# 
# 
# mid = np.nanmax(pdf_avg)/2 
# 
# 
# diff = pdf-mid
# i=len(diff)-1
# while diff[i]<0:
#     i-=1
# p_transition_filter=sum(pdf[0:i])*(bins[1]-bins[0])
# p_transition= (1-n_filter/n_total) + p_transition_filter*n_filter/n_total
# 
# lat_transition = np.percentile(lat_df, p_transition*100)
# 
# 
# pdf, bins= np.histogram(lat_df,bins=int(len(lat_df)/3), density=True)
# dbins=bins[1]-bins[0]
# cdf= np.cumsum(pdf)*dbins
# plt.figure()
# plt.plot(bins[:-1],cdf)
# plt.plot(lat_transition,p_transition,'s')
# plt.xlabel('latency')
# plt.ylabel('CDF')
# =============================================================================


# # cdf_second_der=(pdf[1:]-pdf[0:-1])/(bins[1]-bins[0])
# x=bins[0:-1]+dbins/2

# # pdf_avg = pd.Series(pdf).rolling(4).mean().to_numpy()
# # cdf_2nd_der=(pdf[1:]-pdf[0:-1])/dbins

# cdf_unique, cdf_index = np.unique(cdf, return_index=True)
# x_unique=x[cdf_index]

# cdf_1d=(cdf_unique[1:]-cdf_unique[0:-1])/(x_unique[1:]-x_unique[0:-1])

# cdf_2d=(cdf_1d[1:]-cdf_1d[0:-1])/(x_unique[1:-1]-x_unique[0:-2])
# # cdf_2d_avg = pd.Series(cdf_2d).rolling(14).mean().to_numpy()
# k=cdf_2d/(1+cdf_1d[0:-1]**2)**1.5

# plt.subplot(4,1,1)
# plt.plot(x_unique,cdf_unique,'.')

# plt.subplot(4,1,2)
# plt.plot(x_unique[0:-1], cdf_1d,'.')

# plt.subplot(4,1,3)
# plt.plot(x_unique[0:-2],cdf_2d,'.')
# # plt.plot(cdf_2d_avg,'.')
# plt.subplot(4,1,4)
# plt.plot(x_unique[0:-2],abs(k),'.')

# lat_max=2e4

#%% cdf of latency
# =============================================================================
# plt.figure()
# plt.rcParams['font.size'] = '16'
# plt.hist(df[:,-1],bins=bins, cumulative=True, density=True,histtype='step',range=(0,lat_max))
# plt.hist(train_2000[:,-1],bins=bins, cumulative=True, density=True,histtype='step',range=(0,lat_max))
# # plt.legend(['whole data','2000 samples'])
# plt.xlabel('latency')
# plt.ylabel('CDF')
# plt.title(fileInfo)
# =============================================================================


#%% find transition point
# decide p95

# =============================================================================
# bins=1000
# lat_df=df[:,-1]
# lat_p95 = np.percentile(lat_df, 95)
# ratio=0.95/lat_p95
# 
# pdf, bins= np.histogram(lat_df,bins=bins, density=True)  #,range=(0,lat_max)
# 
# def transitionPoint(angle):
#     diff = pdf-ratio*np.tan(angle)
#     i=len(diff)-1
#     while diff[i]<0:
#         i-=1
#     p_transition=sum(pdf[0:i])*(bins[1]-bins[0])
#     lat_transition = np.percentile(lat_df, p_transition*100)
#     return lat_transition,p_transition
# lat_transition_45,p_transition_45=transitionPoint(np.pi/4)
# lat_transition_30,p_transition_30=transitionPoint(np.pi/6)
# =============================================================================


# =============================================================================
# plt.figure(1)
# plt.plot(lat_transition_45,p_transition_45,'go')
# plt.plot(lat_transition_30,p_transition_30,'ro')
# plt.plot(lat_p95, 0.95,'bs')
# =============================================================================



#%%
# =============================================================================
# # frac=1
# train_data = train_data.sample(frac=1).reset_index(drop=True)
# train_data = train_data.values
# # 31 = number of input features
# train_input = train_data[:,:input_feature]
# train_output = train_data[:,input_feature]
# 
# lat_threshold = np.percentile(train_output, 85)
# 
# print("lat_threshold: ",lat_threshold)
# num_train_entries = int(len(train_output) * 0.80)
# print("num train entries: ",num_train_entries)
# 
# train_Xtrn = train_input[:num_train_entries,:]
# train_Xtst = train_input[num_train_entries:,:]
# train_Xtrn = np.array(train_Xtrn)
# train_Xtst = np.array(train_Xtst)
# 
# =============================================================================
# Classification

# train test train_2000

def classify(lat_list, lat_threshold):
    return np.array(lat_list<lat_threshold).astype(int)

# print("Percent for transition: "+str(p_transition_30))

# lat_threshold=lat_transition_30
# lat_threshold = np.percentile(lat_df, 85)
train_y = classify(train[:,-1], lat_threshold)
test_y = classify(test[:,-1], lat_threshold)

train_X = train[:,0:-1]
test_X = test[:,0:-1]

df_y = classify(df[:,-1], lat_threshold)
df_X = df[:,0:-1]
#%%

# apply DT

# =============================================================================
# param_range = np.arange(1, input_feature, 1)
# train_scores, valid_scores = validation_curve(tree.DecisionTreeClassifier(), df_X, df_y,
#                                               param_name="max_depth",param_range = param_range,
#                                               cv=5)
# plt.figure()
# plt.plot(param_range, np.mean(train_scores,axis=1), label="Training score", color="blue")
# plt.plot(param_range, np.mean(valid_scores,axis=1), label="Cross-validation score", color="red")
# plt.xticks(param_range)
# plt.legend(loc="best")
# plt.xlabel("max_depth")
# plt.ylabel("Accuracy")
# plt.title(fileInfo + '  p_transition:'+str(round(p_transition_30,2)))
# =============================================================================
# plt.savefig('q6d_validation.png')

#%%
# train_sizes=np.arange(200,5000,200)
# train_sizes, train_scores, valid_scores = learning_curve(tree.DecisionTreeClassifier(max_depth=5), 
#                                                           df_X, df_y, train_sizes=train_sizes, cv=5)
# plt.figure()
# plt.plot(train_sizes, np.mean(train_scores,axis=1), label="Training score", color="blue")
# plt.plot(train_sizes, np.mean(valid_scores,axis=1), label="Cross-validation score", color="red")
# plt.legend(loc="best")
# plt.xlabel("train_sizes")
# plt.ylabel("score")
# plt.title(fileInfo)
# plt.savefig('q6d_learning.png')


#%% vary training size

# =============================================================================
# number_of_rows = train_X.shape[0]
# 
# train_sizes=np.arange(200,10000,200)
# 
# accu_list=[]
# for size in train_sizes:
#     accu=0
#     for i in range(10):
#         random_indices = np.random.choice(number_of_rows, size=size, replace=False)
#         sample_X = train_X[random_indices, :]
#         sample_y = train_y[random_indices]
#         # Create Decision Tree classifer object
#         clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
#         
#         # Train Decision Tree Classifer
#         clf = clf.fit(sample_X,sample_y)
#         
#         #Predict the response for test dataset
#         pred_y = clf.predict(test_X)
#         
#         accu+=metrics.accuracy_score(test_y, pred_y)
#         
#     accu_list.append(accu/10)
#     
# plt.figure()
# plt.plot(train_sizes,accu_list)
# plt.xlabel("train sizes")
# plt.ylabel("Accuracy")
# plt.title('total size of training: '+str(number_of_rows) + "  test size: "+str(len(test_y)))
# =============================================================================

#%%
def accu_fp(test_y, pred_y):
    tmp=test_y==pred_y
    accu=sum(tmp)/len(tmp)
    FP=sum((test_y-pred_y)==-1)/len(tmp)
    return accu,FP

#%%
n_DT=np.array([0,1,2,3,4,5,10])
accu_DT=np.zeros(7)
FP_DT=np.zeros(7)
#%%


input_feature=(10)
# Create Decision Tree classifer object
clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)

# Train Decision Tree Classifer
clf = clf.fit(train_X[:,input_feature].reshape(-1, 1),train_y)

#Predict the response for test dataset
pred_y = clf.predict(test_X[:,input_feature].reshape(-1, 1))


# Model Accuracy, how often is the classifier correct?
# print("Accuracy n=0:",metrics.accuracy_score(test_y, pred_y))
accu_DT[0], FP_DT[0]=accu_fp(test_y, pred_y)


#%%
# input_feature=(0,10,11)
input_feature=(0,10,11)
# Create Decision Tree classifer object
clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)

# Train Decision Tree Classifer
clf = clf.fit(train_X[:,input_feature],train_y)

#Predict the response for test dataset
pred_y = clf.predict(test_X[:,input_feature])


accu_DT[1], FP_DT[1]=accu_fp(test_y, pred_y)
#%%
# input_feature=(0,10,11)
input_feature=(0,1,10,11,12)
# Create Decision Tree classifer object
clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)

# Train Decision Tree Classifer
clf = clf.fit(train_X[:,input_feature],train_y)

#Predict the response for test dataset
pred_y = clf.predict(test_X[:,input_feature])


accu_DT[2], FP_DT[2]=accu_fp(test_y, pred_y)
#%%
# input_feature=(0,10,11)
input_feature=(0,1,2,10,11,12,13)
# Create Decision Tree classifer object
clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)

# Train Decision Tree Classifer
clf = clf.fit(train_X[:,input_feature],train_y)

#Predict the response for test dataset
pred_y = clf.predict(test_X[:,input_feature])


accu_DT[3], FP_DT[3]=accu_fp(test_y, pred_y)


#%%
# input_feature=(0,10,11)
input_feature=(0,1,2,3,10,11,12,13,14)
# Create Decision Tree classifer object
clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)

# Train Decision Tree Classifer
clf = clf.fit(train_X[:,input_feature],train_y)

#Predict the response for test dataset
pred_y = clf.predict(test_X[:,input_feature])


# Model Accuracy, how often is the classifier correct?
# print("Accuracy n=4:",metrics.accuracy_score(test_y, pred_y))
accu_DT[4], FP_DT[4]=accu_fp(test_y, pred_y)
#%%
# input_feature=(0,10,11)
input_feature=(0,1,2,3,4,10,11,12,13,14,15)
# Create Decision Tree classifer object
clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)

# Train Decision Tree Classifer
clf = clf.fit(train_X[:,input_feature],train_y)

#Predict the response for test dataset
pred_y = clf.predict(test_X[:,input_feature])


# Model Accuracy, how often is the classifier correct?
# print("Accuracy n=5:",metrics.accuracy_score(test_y, pred_y))
accu_DT[5], FP_DT[5]=accu_fp(test_y, pred_y)

#%%

# Create Decision Tree classifer object
clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)

# Train Decision Tree Classifer
clf = clf.fit(train_X,train_y)

#Predict the response for test dataset
pred_y = clf.predict(test_X)


accu_DT[6], FP_DT[6]=accu_fp(test_y, pred_y)

#%%
# =============================================================================
# from sklearn.tree import export_graphviz
# from six import StringIO  
# # from sklearn.externals.six import StringIO  
# from IPython.display import Image  
# import pydotplus
# 
# dot_data = StringIO()
# export_graphviz(clf, out_file=dot_data,  
#                 filled=True, rounded=True,
#                 special_characters=True,class_names=['0','1'])
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
# graph.write_png('0.png')
# Image(graph.create_png())
# =============================================================================


#%%
# plt.figure()
# plt.plot(df_X[:,10], df_y,'.')   # 1 means quick, 0 means slow
# plt.xlabel('current queue length')
# plt.ylabel('label')

#%%

# =============================================================================
# plt.subplot(2,3,1)
# plt.plot(np.arange(6),accu_n,'-o')
# plt.xlabel('n')
# plt.ylabel('Accuracy')
# =============================================================================
# plt.title('Percent for transition: '+str(round(p_transition_30,4)))
#%%
# =============================================================================
# plt.subplot(2,3,2)
# plt.plot(df_X[:,10], df[:,-1],'.')
# plt.xlabel('current queue length')
# plt.ylabel('latency')
# plt.title(fileInfo)
# =============================================================================
#%%

# =============================================================================
# # plt.figure()
# plt.subplot(2, 3, 3)
# plt.scatter(df_X[:,0], df_X[:,11], s=2, c=df_y, cmap=None)
# 
# plt.subplot(2, 3, 4)
# plt.scatter(df_X[:,1], df_X[:,12], s=2, c=df_y, cmap=None)
# 
# plt.subplot(2, 3, 5)
# plt.scatter(df_X[:,2], df_X[:,13], s=2, c=df_y, cmap=None)
# 
# plt.subplot(2, 3, 6)
# plt.scatter(df_X[:,9], df_X[:,20], s=2, c=df_y, cmap=None)
# =============================================================================



