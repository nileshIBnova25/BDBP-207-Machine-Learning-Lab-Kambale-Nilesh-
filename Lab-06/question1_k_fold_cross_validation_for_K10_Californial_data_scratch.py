#-------------Import-----------------------------------#
#Due to Scratch implementation & use of For loops average time taken for running code is 180sec on 13th gen i5 intel#
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import time
#------------------------------------------------------#
start = time.time()


#======================================================#
#-------------------- set parameters here--------------#
col = 5             #                                  #      # it will be column for y = 4 for disease &/ 5 for disease fluct
alpha = 1e-5        #                                  #      # learning rate
iteration = 1000    #                                  #      # No of iteration for large data if need to increase
tol = 0.001         #                                  #      # Toleration limit for difference between 2 theta
k = 10              #                                  #      # K fold validation of data set
#------------------------------------------------------#
#======================================================#



#---------------------------------------------------------------------------------------#
# 1)Load Data Set


#---------------------------------------------------------------------------------------#
# b) California Housing
def load_data():
    [X,y] = fetch_california_housing(return_X_y=True) # 1)Load Data Set
    return X,y
X1,Y1=load_data()
X1=pd.DataFrame(X1)
Y1=pd.Series(Y1)


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
def hypothesis_fun(x, theta, i2):
    total = 0
    for i1 in range(len(theta)):
        total += x[i2][i1] * theta[i1]
    return total

def deriv_fun(x, y, theta, i3):
    z = 0
    for i2 in range(len(x)):
        z += (hypothesis_fun(x, theta, i2) - y[i2]) * x[i2][i3]
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

#-------------------------R2 SCORE CALCULATION FUNCTION---------------------------------#

def r2score(y_test, y_pred):
    mean_y= sum(y_test) / len(y_test)
    sum_of_square_test = sum((y_test[i] - mean_y) ** 2 for i in range(len(y_test)))
    sum_of_square_res = sum((y_test[i] - y_pred[i]) ** 2 for i in range(len(y_test)))
    return 1 - (sum_of_square_res/sum_of_square_test)

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
            updated_theta = get_theta(theta0,X_train,y_train,alpha,iteration,tol)
            y_pred = y_predict(X_test, updated_theta)
            r2_score = r2score(y_test,y_pred)
            scores.append(r2_score)
            print("#----------------------------------------------------------------------------------#")
            print(f"Fold {i+1} :")
            print(f"Score: {r2_score}")
            print("#----------------------------------------------------------------------------------#")
        valid = sum(scores)/len(scores)
        return valid


    valid_it=k_fold_validation(X1,Y1,tol,alpha,k,iteration)
    print(f"Average validation score of all folds is : {valid_it}")

if __name__ == "__main__":
    main()
#--------------------------------------------------------------------------------------------------------------#
end = time.time()
print(f"{end - start} in sec || in minutes {(end - start)/60} ||")



