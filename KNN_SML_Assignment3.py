#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import math
from sklearn import metrics
from sklearn.preprocessing import scale
from collections import Counter
from scipy.spatial import distance
Xtrain= pd.read_csv('mnist_train_digits.csv',header =None)
Xtest= pd.read_csv('mnist_test_digits.csv', header =None)
x_train = np.array(Xtrain.iloc[:, 1:])
y_train = np.array(Xtrain.iloc[:, 0])
x_test = np.array(Xtest.iloc[:, 1:])
y_test = np.array(Xtest.iloc[:, 0])
y_train = y_train.reshape((y_train.shape[0],1))
y_test = y_test.reshape((y_test.shape[0],1))
class kNN():
    def __init__(self):
        pass

    def fit(self, X, y):
        self.data = X
        self.targets = y

    def euclidean_distance(self, X):
        """
        Computes the euclidean distance between the training data and
        a new input example or matrix of input examples X
        """
        # input: single data point
        if X.ndim == 1:
            l2 = np.sqrt(np.sum((self.data - X)**2, axis=1))

        # input: matrix of data points
        if X.ndim == 2:
            n_samples, _ = X.shape
            l2 = [np.sqrt(np.sum((self.data - X[i])**2, axis=1)) for i in range(n_samples)]

        return np.array(l2)

    def predict(self, X, k=1):
        """
        Predicts the classification for an input example or matrix of input examples X
        """
        # step 1: compute distance between input and training data
        dists = self.euclidean_distance(X)

        # step 2: find the k nearest neighbors and their classifications
        if X.ndim == 1:
            if k == 1:
                nn = np.argmin(dists)
                return self.targets[nn]
            else:
                knn = np.argsort(dists)[:k]
                y_knn = self.targets[knn]
                max_vote = max(y_knn, key=list(y_knn).count)
                return max_vote

        if X.ndim == 2:
            knn = np.argsort(dists)[:, :k]
            y_knn = self.targets[knn]
            if k == 1:
                return y_knn.T
            else:
                n_samples, _ = X.shape
                max_votes = [max(y_knn[i], key=list(y_knn[i]).count) for i in range(n_samples)]
                return max_votes
            
kVals = [1,3,5,10,20,30,40,50,60]
accuracies = []
for k in kVals:
    knn = kNN()
    knn.fit(x_train, y_train)
    predicted_label=[]
    for i in range(x_test.shape[0]):
        predicted_label.append(knn.predict(x_test[i], k))
    i=0
    correct=0
    for i in range(x_test.shape[0]):
        if predicted_label[i] == y_test[i]:
            correct+=1
    accuracies.append((correct/x_test.shape[0])*100)
k = 0
for k in range(len(kVals)):
    print("k=%d, accuracy=%.2f%%" % (kVals[k], accuracies[k] * 100))
i = np.argmax(accuracies)
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],accuracies[i]))
plt.plot(kVals,accuracies)
plt.xlabel(' K ')
plt.ylabel(' Accuracies ')
plt.show()


# In[ ]:





# In[ ]:




