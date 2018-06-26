# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 18:19:53 2018

@author: User
"""

import pandas as pd
import glob
import os
import numpy as np

from sklearn.model_selection import train_test_split


image_dir = 'C:/Users/User/Downloads/Capstone Project/train1/'
bottle_dir = 'C:/Users/User/Downloads/Capstone Project/train1/bottleneck/'
folder = os.listdir(bottle_dir)

filenames = []
filename = []
for f in folder:
    filename = glob.glob(bottle_dir + f + "/*.txt")
    filenames.append(filename)


x_data = np.zeros((10222,2048))
y_data = []
read = np.zeros((1,2048))



count = 0
for f in range(len(filenames)):
    
    for g in range(len(filenames[f])):
        read = pd.read_csv(filenames[f][g],sep=',',header=None)
        x_data[count+g] = read
    count += len(filenames[f])
  
for f in range(len(filenames)):
    
    for g in range(len(filenames[f])):
        
        y_data.append(folder[f])

    
labels = pd.get_dummies(list(y_data))
labels3 = np.dot(labels.values,np.arange(1,121))

X_train, X_test, y_train, y_test = train_test_split(x_data, labels3, test_size=0.20, random_state=42)

#############################################################################################

from sklearn import svm
model = svm.SVC(C = 0.025)

model.fit(X_train,y_train)

pred = model.predict(X_test)

correct = 0
length = len(X_test)

correct = pred == y_test
my_accuracy = (np.sum(correct) / length)*100
print ('svm LR Accuracy %: ', my_accuracy)
#############################################################################################

from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(loss='modified_huber',shuffle = True, random_state = 101)
sgd.fit(X_train, y_train)

y_pred = sgd.predict(X_test)

correct1 = 0
length1 = len(X_test)

correct1 = y_pred == y_test
my_accuracy1 = (np.sum(correct1) / length1)*100
print ('sgd LR Accuracy %: ', my_accuracy1)
##############################################################################################

from sklearn.NaiveBayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)

pred1 = nb.predict(X_test)

correct2 = 0
length2 = len(X_test)

correct2 = pred1 == y_test
my_accuracy2 = (np.sum(correct2) / length2)*100
print ('nb LR Accuracy %: ', my_accuracy2)
############################################################################################

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, y_train)

y_pred1 = knn.predict(X_test)

correct3 = 0
length3 = len(X_test)

correct3 = y_pred1 == y_test
my_accuracy3 = (np.sum(correct3) / length3)*100
print ('knn LR Accuracy %: ', my_accuracy3)
###########################################################################################





