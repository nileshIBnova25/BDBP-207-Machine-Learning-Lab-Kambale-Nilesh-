# 3)Use the above simulated CSV file and implement the following from scratch in Python
      # Read simulated data csv file
      # Form x and y (disease_score_fluct)
      # Write a function to compute hypothesis
      # Write a function to compute the cost
      # Write a function to compute the derivative
      # Write update parameters logic in the main function
#-------------Import-----------------------------------#
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#------------------------------------------------------#



#======================================================#
#-------------------- set parameters here--------------#
col = 5             #                                  #      # it will be column for y = 4 for disese &/ 5 for disese fluct
alpha = 1e-3        #                                  #      # learning rate
iteration = 1000    #                                  #      # No of iteraton for large data if nned to increase
tol = 0.001         #                                  #      # Toleration limit fopr diffrence between 2 theta
#------------------------------------------------------#
#======================================================#



#---------------------------------------------------------------------------------------#
# 1)Load Data Set

df=pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
df.plot()
X=df.iloc[:,0:5]
y=df.iloc[:,col+1]

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
#---------------------------------------------------------------------------------------#

#-------------------------R2 SCORE CALCULATION FUNCTION---------------------------------#

def r2score(y_test, y_pred):
    mean_y= sum(y_test) / len(y_test)
    sum_of_square_test = sum((y_test[i] - mean_y) ** 2 for i in range(len(y_test)))
    sum_of_square_res = sum((y_test[i] - y_pred[i]) ** 2 for i in range(len(y_test)))
    return 1 - (sum_of_square_res/sum_of_square_test)

#---------------------------------------------------------------------------------------#


#---------------------------------------------------------------main function---------------------------------#
def main():
    def get_theta(theta, x, y, alpha, iteration, tol):

        for itr in range(iteration):
            new_theta = [0] * len(theta)

            for i3 in range(len(theta)):
                new_theta[i3] = round(theta[i3] - alpha * deriv_fun(x, y, theta, i3),10)

            # check difference between old and new theta
            diff = max(abs(new_theta[i] - theta[i]) for i in range(len(theta)))

            theta = new_theta  # update theta

            j_theta = cost_fun(x, y, theta)
#            print(f'Iteration {itr}: theta = {theta}, cost = {j_theta}, diff = {diff}')

            # stop if difference is small
            if diff < tol:
                print(f"Stopped early at iteration {itr}")
                #print(len(x_test), len(x_test[0]), len(theta))
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

    y_pred=y_predict(x_test,result)
    r2=r2score(y_test,y_pred)

    print(f"Final theta:,{result},predicted {y_pred}")
    print(f"final r2:{r2}")

if __name__ == "__main__":
    main()
#--------------------------------------------------------------------------------------------------------------#








