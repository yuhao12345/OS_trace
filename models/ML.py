import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn import tree
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import metrics


#%%
input_feature=5

# train_input_path = sys.argv[1]
train_input_path="E:/github/IONet-Models/TRACEPROFILE/sdd/out-rerated-100.0/azure.trace_sub-A.tmp"
print("Train input path: " + train_input_path)

train_data = pd.read_csv(train_input_path, dtype='float32',sep=',', header=None)

#%%
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

X=train_data[:,0:5]
Y=train_y
#%%

# apply DT
param_range = np.arange(1, 5, 1)
train_scores, valid_scores = validation_curve(tree.DecisionTreeClassifier(), X, Y,
                                              param_name="max_depth",param_range = param_range,
                                              cv=5)
plt.figure(0)
plt.plot(param_range, np.mean(train_scores,axis=1), label="Training score", color="blue")
plt.plot(param_range, np.mean(valid_scores,axis=1), label="Cross-validation score", color="red")
plt.legend(loc="best")
plt.xlabel("max_depth")
plt.ylabel("score")
# plt.savefig('q6d_validation.png')

train_sizes=np.arange(200,5000,200)
train_sizes, train_scores, valid_scores = learning_curve(tree.DecisionTreeClassifier(max_depth=3), 
                                                         X, Y, train_sizes=train_sizes, cv=5)
plt.figure(1)
plt.plot(train_sizes, np.mean(train_scores,axis=1), label="Training score", color="blue")
plt.plot(train_sizes, np.mean(valid_scores,axis=1), label="Cross-validation score", color="red")
plt.legend(loc="best")
plt.xlabel("train_sizes")
plt.ylabel("score")
# plt.savefig('q6d_learning.png')

#%%
X_train = X[:num_train_entries,:]
X_test = X[num_train_entries:,:]
y_train = Y[:num_train_entries]
y_test = Y[num_train_entries:]

#%%
# Create Decision Tree classifer object
clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#%%
from sklearn.tree import export_graphviz
from six import StringIO  
# from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())
#%%
# apply logistic regression
param_range = np.arange(1, 1000, 40)
train_scores, valid_scores = validation_curve(LogisticRegression(), X, Y,
                                              param_name="C",param_range = param_range,
                                              cv=5)
plt.figure(2)
plt.plot(param_range, np.mean(train_scores,axis=1), label="Training score", color="blue")
plt.plot(param_range, np.mean(valid_scores,axis=1), label="Cross-validation score", color="red")
plt.legend(loc="best")
plt.xlabel("C")
plt.ylabel("score")
# plt.savefig('q6e_validation.png')


train_sizes=np.arange(200,2000,200)
train_sizes, train_scores, valid_scores = learning_curve(LogisticRegression(C=121), 
                                                         X, Y, train_sizes=train_sizes, cv=5)
plt.figure(3)
plt.plot(train_sizes, np.mean(train_scores,axis=1), label="Training score", color="blue")
plt.plot(train_sizes, np.mean(valid_scores,axis=1), label="Cross-validation score", color="red")
plt.legend(loc="best")
plt.xlabel("train_sizes")
plt.ylabel("score")
# plt.savefig('q6e_learning.png')

#%%
model = LogisticRegression()
model.fit(X_train, np.array(y_train)[:,0])
y_pred=model.predict(X_test)
accuracy = metrics.accuracy_score(np.array(y_test)[:,0], y_pred)
print(accuracy)