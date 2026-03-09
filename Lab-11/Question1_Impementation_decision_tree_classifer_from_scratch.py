#----------------------Imports-----------------------------#
import pandas as pd
import math
from sklearn.datasets import load_iris
from collections import  Counter
#----------------------------------------------------------#

#----------------Loading Data-------------------------------#
iris=load_iris(as_frame=True)
df=iris.frame
X = df.iloc[:,:-1]  # features
y = df.ilic[:,-1]   # target
#----------------------------------------------------------#

#---------------------Entropy Function-------------------------------------#
def entropy(labels):
    if len(labels) == 0:
        return 0
    counts = Counter(labels)
    e=0
    for count in counts.values():
        prb = count/len(labels)
        if prb > 0:
            e-= prb * math.log2(prb)
    return e
#----------------------------------------------------------#


#--------------------wighted entropy--------------------------------------#
def weight_entropy(X_column,y):
    total = len(y)
    weighted_e = 0
    for val in X_column.unique():
        subset_labels =  y[X_column ==val]
        weighted_e+= (len(subset_labels/total) ) * entropy(subset_labels)

    return weighted_e
#-------------------------------------------------------------------------#

#--------------------Information gain-----------------------------------------------------#
def information_gain(X,y):
    parent_e = entropy(y)
    igs={}
    cols = X.shape[1]
    for i in range(cols):
        col = X.iloc[:,i]
        igs[i] = parent_e - (weight_entropy(col,y))
    return igs
#-------------------------------------------------------------------------#


