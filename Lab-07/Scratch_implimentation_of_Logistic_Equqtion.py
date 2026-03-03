#-------------Import-----------------------------------#
import numpy as np
import pandas as pd
import time
import math
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
start_time = time.time()


#======================================================#
#-------------------- set parameters here--------------#
delta = 0.01        #                                  #      # DELTA new parameter
alpha = 1e-3        #                                  #      # learning rate
iteration = 1000    #                                  #      # No of iteraton for large data if nned to increase
tol = 0.0000000001  #                                  #      # Toleration limit fopr diffrence between 2 theta
#------------------------------------------------------#
#======================================================#



#-------------------------------------------------LOADING-DATA-SET----------------------------------------------------#
df=pd.read_csv('data.csv')
print(df.info)
print(df.head())
print(df.describe())
print(df.columns)

#------------------------------------Droping-Unnamed-Column-NAN------------------------#
df.drop(columns=['Unnamed: 32','id'],axis=1,inplace=True)
#--------------------------------------------------------------------------------------#

#---------------------------------------Encoding-M-as-1-&-B-as-0-------------------------------------#
df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})
print(df.columns)
#----------------------------------------------------------------------------------------------------#

#-----------------------------DROPING-DIAGNOSIS-COLUMN--&-SAVING-IT-IN-TO-X-&-y----------------------#
X=df.drop('diagnosis',axis=1)
y=df['diagnosis']
print(df['diagnosis'])
#-------------------------------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------#

#2) Divide it into train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

n,d=X_train.shape
#---------------------------------------------------------------------------------------#
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


#---------------------------------------------------------------------------------------#
def add_intercept(X_train):
    x0=np.ones((X_train.shape[0],1))
    X_train_add=np.hstack((x0,X_train))
    return X_train_add


def cr_theta(X_train_inc):
    theta=[0]*X_train_inc.shape[1]
    return theta

X_train_inc=(add_intercept(X_train))
theta=cr_theta(X_train_inc)
X_test_inc=(add_intercept(X_test))


# changing data frame to list
x=X_train_inc.tolist()
y=y_train.tolist()
x_test=X_test_inc.tolist()
y_test=y_test.tolist()

#----------------------------------------------------------------------------------------#



#----------------------------------------------------------------------------------------#
# def sigmoid_function(x):
#     return (1/(1+math.exp(-x)*0.1))


def sigmoid_function(x):
    if x >= 0:
        return 1/(1+math.exp(-x))
    else:
        exp_x=math.exp(x)
        return exp_x/(1+exp_x)



def hypothesis_fun(x, theta, i2):
    sum = 0
    for i1 in range(len(theta)):
        sum += x[i2][i1] * theta[i1]
    total = sigmoid_function(sum)
    return total


def deriv_fun(x, y, theta, i3):
    z = 0
    for i2 in range(len(x)):
        z += (y[i2] - hypothesis_fun(x, theta, i2) ) * x[i2][i3]
    return z


def cost_fun(x,y,theta):
    J_of_theta = 0
    for i in range(len(x)):
        J_of_theta += (hypothesis_fun(x, theta, i) - y[i]) ** 2
    return J_of_theta
#---------------------------------------------------------------------------------------#

# #-------------------------R2 SCORE CALCULATION FUNCTION---------------------------------#
#
# def r2score(y_test, y_pred):
#     mean_y= sum(y_test) / len(y_test)
#     sum_of_square_test = sum((y_test[i] - mean_y) ** 2 for i in range(len(y_test)))
#     sum_of_square_res = sum((y_test[i] - y_pred[i]) ** 2 for i in range(len(y_test)))
#     return 1 - (sum_of_square_res/sum_of_square_test)
#
# #---------------------------------------------------------------------------------------#

#---------------------Accuracy Calculation for Logistic regression----------------------#

def logi_accuracy(y_test, y_pred):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for i in range(len(y_test)):

        if y_test[i] == 1 and y_pred[i] == 1:
            true_positive += 1
        elif y_test[i] == 0 and y_pred[i] == 0:
            true_negative += 1
        elif y_test[i] == 1 and y_pred[i] == 0:
            false_negative += 1
        else:
            false_positive +=1
    accuracy = (true_positive+true_negative)/(false_positive+false_negative+true_positive+true_negative)

    return  accuracy





#---------------------------------------------------------------------------------------#

#---------------------------------------------------------------main function---------------------------------#
def main():
    def get_theta(theta, x, y, alpha, iteration, tol):

        for itr in range(iteration):
            new_theta = [0] * len(theta)

            for i3 in range(len(theta)):
                new_theta[i3] = theta[i3] + alpha * deriv_fun(x, y, theta, i3)

            # check difference between old and new theta
            diff = max(abs(new_theta[i] - theta[i]) for i in range(len(theta)))

            theta = new_theta  # update theta

            j_theta = cost_fun(x, y, theta)
#            print(f'Iteration {itr}: theta = {theta}, cost = {j_theta}, diff = {diff}')

            # stop if difference is small
            if diff < tol:
                print(f"Stopped early at iteration {itr}")

                break
        return theta

    result = get_theta(theta, x, y, alpha, iteration, tol)
    print(len(x_test),len(x_test[0]),len(theta))

    def y_predict(a, b):
        rows_a = len(a)
        cols_a = len(a[0])

        if cols_a != len(b):
            raise ValueError("Number of columns of a must match size of vector b")

        result = [0] * rows_a

        for i in range(rows_a):
            for k in range(cols_a):
                result[i] += a[i][k] * b[k]

        return result

    raw_y_pred = y_predict(x_test, result)
    y_pred=[1 if sigmoid_function(z) > 0.5 else 0 for z in raw_y_pred]
    accuracy=logi_accuracy(y_test,y_pred)


    print(f"Final theta:,{result},predicted {y_pred}")
    print(f"final accuracy : {accuracy} /")
    print("")
    print(sum(y_pred))
    print(sum(y_test))

if __name__ == "__main__":
    main()
#--------------------------------------------------------------------------------------------------------------#
