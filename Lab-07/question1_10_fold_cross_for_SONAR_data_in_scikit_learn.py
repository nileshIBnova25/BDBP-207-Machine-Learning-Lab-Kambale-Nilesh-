#----------IMPORT--------------------------------------------#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import math
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report
#-----------------------------------------------------------#


#======================================================#
#-------------------- set parameters here--------------#
col = 5             #                                  #      # it will be column for y = 4 for disease &/ 5 for disease fluct
alpha = 1e-3        #                                  #      # learning rate
iteration = 1000    #                                  #      # No of iteration for large data if need to increase
tol = 0.001         #                                  #      # Toleration limit for difference between 2 theta
k = 10                 #                                  #      # K fold validation of data set
#------------------------------------------------------#
#======================================================#


#---------------------LOADING-DATA--------------------------#

df=pd.read_csv("sonar.csv" , header=None )
print(df.describe)
print(df.info)
print(df.columns)
#---------------------------PLOTING-HISTOGRAM--FOR-ALL-FEATURES--------------------------------------#
#df.hist(figsize=(30,30))
plt.show()
#----------------------------------------------------------------------------------------------------#
#---------------------------------Encoding-M-as-1-&-B-as-0-------------------------------------------#
df[60]=df[60].map({'M':0,'R':1})
#----------------------------------------------------------------------------------------------------#
print(df.describe)

#-----------------------------DROPING-DIAGNOSIS-COLUMN--&-SAVING-IT-IN-TO-X-&-y----------------------#
X=df.drop(60,axis=1)
y=df[60]
#----------------------------------------------------------------------------------------------------#

#--------------------------------Randomizing-The-Sample----------------------------------------------#
X1,Y1=shuffle(X,y,random_state=42)
#----------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------#
def add_intercept(X_train):
    x0=np.ones((X_train.shape[0],1))
    X_train_add=np.hstack((x0,X_train))
    return X_train_add


def cr_theta(X_train_inc):
    theta=[0]*X_train_inc.shape[1]
    return theta


#----------------------------------------------------------------------------------------#
scaler = StandardScaler()

def k_fold_creation(X,Y,k,i):

    start_row=round(i*len(X)/k)                               # This are the lines which will do k fold validation

    if round((i+1)*len(X)/k) > len(X):
        end_row=len(X)
    else :
        end_row=round((i+1)*len(X)/k)
    print(start_row,end_row)

    index = X.index[start_row: end_row]

    scaler = StandardScaler()

    X_pre_list_train = X.drop(index)
    X_pre_scaled_train = scaler.fit_transform(X_pre_list_train)
    X_inc_train = add_intercept(X_pre_scaled_train)
    X_train = X_inc_train.tolist()
    y_train = (Y.drop(index)).values.tolist()

    X_pre_test = (X.iloc[start_row : end_row])
    y_test = (Y.iloc[start_row : end_row]).values.tolist()
    X_scaled_test = scaler.transform(X_pre_test)
    X_intercept_tset= add_intercept(X_scaled_test)
    X_test = X_intercept_tset.tolist()

    theta = cr_theta(X_inc_train)

    return X_train,y_train,X_test,y_test,theta

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

#---------------------------------------------------------------------------------------#


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

#-----------------------------------------------Get-Parametrized-theta----------------------------------------#
def get_theta(theta, x, y, alpha, iteration, tol):

    for itr in range(iteration):
        new_theta = [0] * len(theta)

        for i3 in range(len(theta)):
            new_theta[i3] = round(theta[i3] - alpha * deriv_fun(x, y, theta, i3),10)

        # check difference between old and new theta
        diff = max(abs(new_theta[i] - theta[i]) for i in range(len(theta)))

        theta = new_theta  # update theta

        j_theta = cost_fun(x, y, theta)
#       print(f'Iteration {itr}: theta = {theta}, cost = {j_theta}, diff = {diff}')

        # stop if difference is small
        if diff < tol:
            print(f"Stopped early at iteration {itr}")
            #print(len(x_test), len(x_test[0]), len(theta))
            break
    return theta
#-------------------------------------------------------------------------------------------------------------#


#---------------------------------------------------------------main function---------------------------------#
def main():
    def k_fold_validation(X1,Y1,tol,alpha,k,iteration):
        scores=[]
        for i in range(k):
            X_train,y_train,X_test,y_test,theta0 = k_fold_creation(X1,Y1,k,i)
          #  updated_theta = get_theta(theta0,X_train,y_train,alpha,iteration,tol)
            logistic_model = LogisticRegression(max_iter=1000)
            logistic_model.fit(X_train, y_train)
            y_pred = logistic_model.predict(X_test)

            #raw_y_pred = y_predict(X_test, updated_theta)
            #y_pred = [1 if sigmoid_function(z) > 0.5 else 0 for z in raw_y_pred]
            print( f'{y_test} {y_pred}' )
            accuracy = logi_accuracy(y_test, y_pred)
            #scores.append(accuracy)
            # ----------------------------------------------------------------------------------------------------#
            accuracy = accuracy_score(y_test, y_pred)
            print('Accuracy: {:.2f} %'.format(accuracy * 100))
            # ----------------------------------------------------------------------------------------------------#
            confusion_matrix(y_test, y_pred)
            # ----------------------------------------------------------------------------------------------------#
            print(classification_report(y_test, y_pred))
            scores.append(accuracy)
            # ---------------------------------------------------------------------------------------------------
            print("#----------------------------------------------------------------------------------#")
            print(f"Fold {i+1} :")
            print(f"Score: {accuracy}")
            print("#----------------------------------------------------------------------------------#")
        valid = sum(scores)/len(scores)
        return valid


    valid_it=k_fold_validation(X1,Y1,tol,alpha,k,iteration)
    print(f"Average validation score of all folds is : {valid_it}")




if __name__ == "__main__":
    main()
#--------------------------------------------------------------------------------------------------------------#