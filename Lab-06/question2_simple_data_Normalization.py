#-------------Import-----------------------------------#
import numpy as np
import pandas as pd
from orca.punctuation_settings import infinity
from sklearn.preprocessing import StandardScaler
import random
import sys
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
#------------------------------------------------------#
start = time.time()


#======================================================#
#-------------------- set parameters here--------------#
col = 5             #                                  #      # it will be column for y = 4 for disease &/ 5 for disease fluct
alpha = 1e-4        #                                  #      # learning rate
iteration = 1000    #                                  #      # No of iteration for large data if need to increase
tol = 0.001         #                                  #      # Toleration limit for difference between 2 theta
k = 10              #                                  #      # K fold validation of data set
#------------------------------------------------------#
#======================================================#



#---------------------------------------------------------------------------------------#
# 1)Load Data Set

df=pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')


#----------------------------------FUNCTION-FOR-NORMALIZATION-OF-DATA---------------------------------------------------#
def norm_data(df):
    X_df = df.values.tolist()
    print(len(X_df[1]))
    a=0
    for i in range(len(X_df[1])):
        col_values=[X_df[j][i] for j in range(len(X_df[1]))]
        min_val=min(col_values)
        max_val=max(col_values)

        if max_val - min_val != 0:
            for k in range(len(df)):
                X_df[k][i]= (X_df[k][i]-min_val) / (max_val-min_val)
            a=a+1
        else:
            for k  in range(len(df)):
                X_df[k][i] = 0
            a=a+1

    return X_df
#-----------------------------------------------------------------------------------------------------------------------#
res=norm_data(df)
print(res)

# def norm_data(df,x):
#     x = x.values.tolist()
#     X_df = df.values.tolist()
#     for i in range(len(X_df[1])):   # loop over columns
#         min_val = X_df[0][i]
#         max_val = X_df[0][i]
#
#         for j in range(len(x)):      # loop over rows
#             if X_df[j][i] > max_val:
#                 max_val = X_df[j][i]
#             if X_df[j][i] < min_val:
#                 min_val = X_df[j][i]
#
#         if max_val - min_val != 0:
#             for k in range(len(x)):
#                 x[k][i] = (x[k][i] - min_val) / (max_val - min_val)
#         else:
#             for k in range(len(x)):
#                 x[k][i] = 0
#     return x

#     for i in range(len(X_df[1])):   # loop over columns
#         min_val = X_df[0][i]
#         max_val = X_df[0][i]
#
#         for j in range(len(x)):      # loop over rows                     # here mistake was len(x) this will get min & max for sample only insted of len(X_df)
#             if X_df[j][i] > max_val:
#                 max_val = X_df[j][i]
#             if X_df[j][i] < min_val:
#                 min_val = X_df[j][i]

# for i in range(len(X_df[1])):
#     col_values = [X_df[j][i] for j in range(len(X_df[1]))]
#     min_val = min(col_values)
#     max_val = max(col_values)